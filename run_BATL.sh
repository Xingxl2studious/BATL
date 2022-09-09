CUDA_VISIBLE_DEVICES=0 python3 train_BATL.py \
    --lr 1e-4 \
    --batch_size 32 \
    --max_train_epochs 500 \
    --src_lang 'sq' \
    --device_num '0' \
    --output 'checkpoint4' \
    --seed 1 \
    --HRL 'es' \
    --layer_num 1 \
    --embedding "mean"

# nohup sh run_BATL.sh > sq4.log 2>&1 &