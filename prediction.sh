CUDA_VISIBLE_DEVICES=0 python prediction.py \
                       --dataset 'kitti' \
                       --batchsize 4 \
                       --workers_test 4 \
                       --model MSNet2D \
                       --load_model './Vitis/build/float_model/model.pth'