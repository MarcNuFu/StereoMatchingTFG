echo "Quantization script mode: calib - quantize"
#CUDA_VISIBLE_DEVICES=1 
python -u quantize.py \
          --model DispNet \
          --dataset 'kitti' \
          -d /workspace/Vitis/build \
          --workers_test 1 \
          --batchsize 1 \
          --maxdisp 192 \
          --pth_name Dispnet-test_old \
          --quant_mode calib \
          2>&1 | tee /workspace/Vitis/build/logs/quant_calib.log

echo "Quantization script mode: test - evaluate quantized model"
#CUDA_VISIBLE_DEVICES=1 
python -u quantize.py \
        --model DispNet \
        --dataset 'kitti' \
        -d /workspace/Vitis/build \
        --workers_test 1 \
        --maxdisp 192 \
        --pth_name Dispnet-test_old \
        --quant_mode test  \
        2>&1 | tee /workspace/Vitis/build/logs/quant_test.log