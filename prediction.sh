CUDA_VISIBLE_DEVICES=0 python prediction.py \
                       --dataset 'kitti' \
                       --batchsize 1 \
                       --workers_test 1 \
                       --model DispNet \
                       --load_model './Vitis/build/float_model/DispNetKITTI.pth'