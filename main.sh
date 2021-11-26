python main.py \
       --output_filename 'backbone' \
       --path_dataset './dataset/kitti2015v2/' \
       --batchsize 1 \
       --epochs 10 \
       --workers_train 1 \
       --workers_test 1 \
       --epochs_printed 2 \
       --learnrate 0.00001 \
       --model AutoencoderBaseline