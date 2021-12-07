CUDA_VISIBLE_DEVICES=0 python prediction.py \
                       --dataset 'kitti' \
                       --batchsize 8 \
                       --workers_test 4 \
                       --model AutoencoderBackbone \
                       --load_model './Vitis/build/float_model/backbone_kitti_model.pth'