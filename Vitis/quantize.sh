echo "Quantization script mode: calib - quantize"
python -u quantize.py \
          --model AutoencoderBaseline \
          --dataset 'kitti' \
          -d /workspace/Vitis/build \
          --workers_test 1 \
          --batchsize 4 \
          --quant_mode calib \
          2>&1 | tee /workspace/Vitis/build/logs/quant_calib.log

echo "Quantization script mode: test - evaluate quantized model"
python -u quantize.py \
        --model AutoencoderBaseline \
        --dataset 'kitti' \
        -d /workspace/Vitis/build \
        --workers_test 1 \
        --quant_mode test  \
        2>&1 | tee /workspace/Vitis/build/logs/quant_test.log