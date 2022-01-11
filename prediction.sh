CUDA_VISIBLE_DEVICES=8 python prediction.py \
                       --dataset 'kitti' \
                       --batchsize 1 \
                       --workers_test 1 \
                       --model DispNetV2 \
                       --load_model './Vitis/build/float_model/DispNetV2KITTI_best.pth'