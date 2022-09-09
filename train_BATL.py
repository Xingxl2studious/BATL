import os 
import torch
import argparse
import logging
import shutil
from tqdm import tqdm
import numpy as np
import sentencepiece as spm
# from data.data import *
from data import GANDataSet, MLMDataSet, iterator
from source.model import MLMModel1, Discriminator, Model_for_HRL
from transformers import AdamW, AutoTokenizer
import torch.utils.data as Data
import random
from torch import nn

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding", type=str, default="mean", choices=["standard", "mean"]
    )
    parser.add_argument(
        "--device_num", type=str, default="0"
    )
    parser.add_argument(
        "--src_lang", type=str, default="el"
    )
    parser.add_argument(
        "--HRL", type=str, default="de"
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
    parser.add_argument("--device_no", type=float, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_train_epochs", type=int, default=100)

    args = parser.parse_args()
    return args

def setup_seed(seed):
    print('seed is: ', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = parse_args()
    device_no = args.device_num
    device = torch.device('cuda:' + device_no)
    
    print(f"layers is:{args.layer_num}")
    setup_seed(int(args.seed))

    src_lang = args.src_lang
    HRL_lang = args.HRL
    
    output_dir = 'checkpoint/'+src_lang+'/'+args.output+'/' + str(args.layer_num) +'layer/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger()
    fh = logging.FileHandler(output_dir + 'result.log')
    formats = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formats)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)

    

    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{HRL_lang}-en")
    sp = spm.SentencePieceProcessor(model_file=f'data/{src_lang}/{src_lang}+en_{HRL_lang}_tokenizer/spm.model')
    sp.bos_id=-1
    sp.unk_id=tokenizer.unk_token_id
    sp.eos_id=tokenizer.eos_token_id
    sp.pad_id=tokenizer.pad_token_id
    sp.vocab_size = 100000

    model_de = Model_for_HRL.from_pretrained(f'Helsinki-NLP/opus-mt-{args.HRL}-en')
    model = MLMModel1.from_pretrained(f'Helsinki-NLP/opus-mt-{args.HRL}-en', sp)
    dis = Discriminator(model.config)
    model.encoder.embed_tokens = torch.load(f'embeddings/{src_lang}_{HRL_lang}_pytorch_model_concat_emb.bin')
    model.fc2.weight = model.encoder.embed_tokens.weight

    model_de = model_de.to(device)
    model = model.to(device)
    dis = dis.to(device)

    config = model.config
    max_length = config.max_position_embeddings

    for k, v in model.named_parameters():
        v.requires_grad = False
    for i in range(args.layer_num):
        for k, v in model.encoder.layers[i].named_parameters():
            v.requires_grad = True
    for k, v in model.shared.named_parameters():
        v.requires_grad = True
    for k, v in model.encoder.embed_tokens.named_parameters():
        v.requires_grad = True
    for k, v in model.fc2.named_parameters():
        v.requires_grad = True
    for k, v in model.linear.named_parameters():
        v.requires_grad = True

    batch_size = args.batch_size
    epochs = args.max_train_epochs

    criterion = nn.CrossEntropyLoss()
    criterion_GAN_D = nn.CrossEntropyLoss()
    criterion_GAN_G = nn.CrossEntropyLoss()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer_model = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-5)
    optimizer_dis = torch.optim.Adam(dis.parameters(), lr=1e-5)

    steps = 0
    steps_GAN = 0
    best_bleu = 0
    min_loss = 100
    break_signal = False

    for epoch in range(epochs):
        
        print(" GANing...")
        
        dataset1 = torch.load(f'data/{src_lang}/{src_lang}_GAN.pkl')
        loader_GAN_low = Data.DataLoader(dataset1, batch_size, True)
        it_GAN_low = iterator(loader_GAN_low)
        dataset3 = torch.load(f'data/{src_lang}/{args.HRL}_GAN.pkl')
        loader_GAN_high = Data.DataLoader(dataset3, batch_size, True)
        it_GAN_high = iterator(loader_GAN_high)
        max_value = max(len(loader_GAN_high), len(loader_GAN_low))
        num = max_value//4

        for batch_num in tqdm(enumerate(range(num)), total=num, desc="Iteration"):
            steps_GAN += 1
            
            # GAN discriminator
            model.eval()
            model_de.eval()
            dis.train()
            dis.zero_grad()

            data = it_GAN_high.get_batch()
            input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            hidden = model_de(input_ids=input_ids, attention_mask=attention_mask, hidden_layer=args.layer_num)
            result = dis(hidden, attention_mask)
            loss_real_D = criterion_GAN_D(result, labels.view(batch_size))
            loss_real_D.backward()
            
            data = it_GAN_low.get_batch()
            input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            hidden = model(input_ids=input_ids, attention_mask=attention_mask, hidden_layer=args.layer_num)
            result = dis(hidden, attention_mask)
            loss_fake_D = criterion_GAN_D(result, labels.view(batch_size))
            loss_fake_D.backward()
            optimizer_dis.step()

            # GAN generator
            dis.eval()
            model_de.eval()
            model.train()
            model.zero_grad()

            data = it_GAN_low.get_batch()
            input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            hidden = model(input_ids=input_ids, attention_mask=attention_mask, hidden_layer=args.layer_num)
            result = dis(hidden, attention_mask)
            labels = torch.zeros_like(labels)
            loss_fake_G = criterion_GAN_G(result, labels.view(batch_size))
            loss_fake_G.backward()
            # loss = (loss_fake_D + loss_real_D)/2 
            # loss.backward()
            optimizer_model.step()


        print(" MLMing...")
        dataset2 = torch.load(f'data/{src_lang}/{src_lang}_MLM.pkl')
        loader_MLM = Data.DataLoader(dataset2, batch_size, True)
        it_MLM = iterator(loader_MLM)
        num = len(loader_MLM)//2
        for batch_num, _ in tqdm(enumerate(range(num)), total=num, desc="Iteration"):
            steps += 1
            dis.eval()
            model_de.eval()
            model.train()
            data = it_MLM.get_batch()
            input_ids = data[0].to(device) 
            masked_tokens = data[1].to(device) 
            masked_pos = data[2].to(device)
            attention_mask =  data[3].to(device)
            model.zero_grad()

            logits_lm = model(input_ids=input_ids, masked_pos=masked_pos, attention_mask=attention_mask)
            loss_lm = criterion(logits_lm.view(-1, sp.vocab_size), masked_tokens.view(-1)-1)
            loss_lm = (loss_lm.float()).mean()
            loss = loss_lm
            loss.backward()
            optimizer_model.step()

            if (batch_num + 1) % 1000 == 0:
                print("batch_num is :", batch_num ," loss is :" , loss)
                logger.debug(' steps is {}, Loss is: {}'.format(steps, loss))

            if steps == 150000:
                output_2 = 'checkpoint/'+src_lang+'/'+args.output+'/' + str(args.layer_num) +'layer/'+ "checkpoint_150k"
                if os.path.isdir(output_2):
                    shutil.rmtree(output_2, True)
                if not os.path.exists(output_2):
                    os.makedirs(output_2)
                output_dir = output_2 + '/' + 'checkpoint-best.pkl'
                logger.debug("  Saving 150k model checkpoint to %s", output_dir)
                print(" Saving 150k model checkpoint to %s", output_dir)
                print(" steps:   ", steps,";  loss:", loss)
                checkpoint = {
                'net': model.state_dict(),
                'optimizer': optimizer_model.state_dict(),
                'epoch': epoch,
                'batch': batch_num,
                'loss': loss
                }
                torch.save(checkpoint, output_dir)
                break_signal = True
                break
                
        if break_signal:
            break


