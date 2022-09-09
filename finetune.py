import os
import torch
import random
import argparse
import numpy as np
from source.model import MLMModel1
from source.Marian.modeling_marian import MarianMTModel
import sentencepiece as spm
from data import Seq2SeqDataCollator
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer, 
    AutoTokenizer
)

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
        "--embedding", type=str, default="mean", choices=["standard", "mean"]
    )
    parser.add_argument(
        "--HRL", type=str, default="es"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--freeze_decoder",
        type=int,
        default=0
    )
    args = parser.parse_args()
    return args

def setup_seed(seed):
    print('seed is: ', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print('setup seed')


print("------------")
args = parse_args()
print(f'lang: {args.src_lang}')
print(f'output: {args.output}')
print(f'batch_size: {args.batch_size}')
src_lang = args.src_lang
HRL_lang = args.HRL
tgt_lang = "en"
device = torch.device('cuda:0')
setup_seed(int(args.seed))

# load tokenizer
pretrained_tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{HRL_lang}-en")
sp = spm.SentencePieceProcessor(model_file=f"data/{src_lang}/{src_lang}+{tgt_lang}_{HRL_lang}_tokenizer/spm.model")
sp.bos_id=-1
sp.unk_id=pretrained_tokenizer.unk_token_id
sp.eos_id=pretrained_tokenizer.eos_token_id
sp.pad_id=pretrained_tokenizer.pad_token_id
sp.vocab_size = 100000

# load model
model = MLMModel1.from_pretrained(f'Helsinki-NLP/opus-mt-{HRL_lang}-en', sp)
model.encoder.embed_tokens = torch.load(f'embeddings/{src_lang}_{HRL_lang}_pytorch_model_concat_emb.bin')
model.load_state_dict(torch.load(f'checkpoint/{src_lang}/{args.output}/{args.layer_num}layer/checkpoint_150k/checkpoint-best.pkl', map_location={'cuda:1':'cpu','cuda:0':'cpu','cuda:2':'cpu','cuda:3':'cpu','cuda:4':'cpu','cuda:5':'cpu','cuda:6':'cpu','cuda:7':'cpu','cuda:8':'cpu','cuda:9':'cpu'})['net'])
translation_model = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{HRL_lang}-en')
for i in range(args.layer_num):
    translation_model.model.encoder.layers[i].load_state_dict(model.encoder.layers[i].state_dict())
translation_model.model.encoder.embed_tokens = torch.nn.Embedding(sp.vocab_size, translation_model.config.d_model)
translation_model.model.encoder.embed_tokens.load_state_dict(model.encoder.embed_tokens.state_dict())
translation_model = translation_model.to(device)

# load data
tokenized_dataset_dict = torch.load(f'data/{src_lang}/{src_lang}_{tgt_lang}_tokenized_dataset_dict_{HRL_lang}.pkl')

if args.freeze_decoder == 1:
    modules = [translation_model.model.decoder, translation_model.lm_head, translation_model.model.shared]
else:
    modules = [translation_model.lm_head, translation_model.model.shared]
for module in modules:
    for name, param in module.named_parameters():
        param.requires_grad = False

checkpoint_path = f'checkpoint/{src_lang}/{args.output}/{str(args.layer_num)}layer/param_result/'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoint_path,
    evaluation_strategy="steps",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=0.01,
    save_total_limit=15,
    num_train_epochs=200,
    seed=int(args.seed),
    load_best_model_at_end=True,
    predict_with_generate=True,
    remove_unused_columns=True,
    fp16=True,
    gradient_accumulation_steps=2,
    eval_steps=500,
    warmup_steps=100,
    dataloader_pin_memory=False,
)
max_source_length = 128
max_target_length = max_source_length
pad_token_id = sp.pad_id
data_collator = Seq2SeqDataCollator(max_source_length, pad_token_id, pretrained_tokenizer.pad_token_id)
callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]

trainer = Seq2SeqTrainer(
    model=translation_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset_dict["train"],
    eval_dataset=tokenized_dataset_dict["dev"],
    callbacks=callbacks,
)
print('training')
trainer_output = trainer.train()


