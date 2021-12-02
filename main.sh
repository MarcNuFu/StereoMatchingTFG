CUDA_VISIBLE_DEVICES=0 python main.py \
       --output_filename 'baseline' \
       --dataset 'kitti' \
       --batchsize 1 \
       --epochs 10 \
       --workers_train 8 \
       --workers_test 4 \
       --epochs_printed 3 \
       --learnrate 0.0001 \
       --model AutoencoderBackbone