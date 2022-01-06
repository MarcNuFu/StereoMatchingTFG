CUDA_VISIBLE_DEVICES=0 python train.py \
                       --dataset 'sceneflow' \
                       --batchsize 32 \
                       --total_epochs 40 \
                       --start_epoch 0 \
                       --workers_train 8 \
                       --workers_test 8 \
                       --learnrate  0.0001 \
                       --model DispNet \
                       --logdir 'tensorboard/train2' \
                       --pth_name 'DispNetSceneFlow2' \
                       --maxdisp 192