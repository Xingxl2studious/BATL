

CUDA_VISIBLE_DEVICES=0 python3 finetune.py \
    --batch_size 64 \
    --seed  1\
    --src_lang "sq" \
    --output "checkpoint2" \
    --embedding "mean" \
    --HRL "es"\
    --freeze_decoder 1\
    --layer_num 1 \
    --lr 5e-5 

CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --batch_size 64 \
    --seed  1\
    --src_lang "sq" \
    --output "checkpoint2" \
    --embedding "mean" \
    --HRL "es"\
    --layer_num 1
    
# nohup sh run_finetune.sh > finetuning_sq2.log 2>&1