
import os
import argparse
from source.evaluate import evaluate_parallel, bleu

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_lang", type=str, default="sq"
    )
    parser.add_argument(
        "--output",
        type=str,
        default='checkpoint1'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--layer_num",
        type=int,
        default=1
    )
    parser.add_argument(
        "--embedding", type=str, default="standard", choices=["standard", "mean"]
    )
    
    parser.add_argument(
        "--metric", type=str, default="loss", choices=["loss", "bleu"]
    )
    parser.add_argument(
        "--HRL", type=str, default="de"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--is_contrast", action='store_true')

    args = parser.parse_args()
    return args


args = parse_args()
src_lang = args.src_lang
HRL_lang = args.HRL
output = args.output
layer_num = args.layer_num

checkpoint_path = f'checkpoint/{src_lang}/{args.output}/{str(args.layer_num)}layer/param_result/'

device_num = "0"
if 'result.txt' in os.listdir(checkpoint_path):
    print('already have result.txt!!!')
else:
    with open(checkpoint_path+'/result.txt', 'w') as f:
        pass
    for checkpoint in os.listdir(checkpoint_path):
        if 'checkpoint-' in checkpoint:
            print(checkpoint)
            PATH = f'{checkpoint_path}/{checkpoint}'
            # dev
            if 'pytorch_model.bin-prediction-dev' not in os.listdir(checkpoint_path + '/' + checkpoint):
                print(PATH)
                b = evaluate_parallel(checkpoint_path=PATH, src_lang=src_lang, HRL_lang=HRL_lang, test_or_dev='dev', device_num=device_num)
            else:
                b = bleu(PATH + "/pytorch_model.bin", 'dev')
                print(b)
            # test
            if 'pytorch_model.bin-prediction-test' not in os.listdir(checkpoint_path + '/' + checkpoint):
                print(PATH)
                b1 = evaluate_parallel(checkpoint_path=PATH, src_lang=src_lang, HRL_lang=HRL_lang, test_or_dev='test', device_num=device_num)
            else:
                b1 = bleu(PATH + "/pytorch_model.bin", 'test')
                print(b1)
            with open(checkpoint_path+'/result.txt', 'a+') as f:
                f.write(checkpoint+'\t'+'dev: '+str(b)+ '\t test:'+str(b1)+ '\n')