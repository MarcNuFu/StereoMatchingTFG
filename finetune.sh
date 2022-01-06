CUDA_VISIBLE_DEVICES=0 python train.py \
                       --dataset 'kitti' \
                       --batchsize 32 \
                       --total_epochs 400 \
                       --start_epoch 0 \
                       --workers_train 8 \
                       --workers_test 8 \
                       --learnrate 0.0001 \
                       --model DispNet \
                       --logdir 'tensorboard/finetune' \
                       --pth_name 'DispNetKITTI' \
                       --maxdisp 192 \
                       --load_model './Vitis/build/float_model/DispNetSceneFlow.pth'
