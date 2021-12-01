# StereoMatchingTFG

## Entrenar red y generación pth
Se genera model.pth en el directorio ./Vitis/build/float_model
```python
sh main.sh
```

## Creación xmodel de la red para ejecutar con Alveo U50 (Vitis Ai)
Se genera el modelo cuantizado en el directorio ./Vitis/build/quant_model y se compila para poder ejecutar en Alveo U50
```python
./docker_run.sh xilinx/vitis-ai-cpu:latest
conda activate vitis-ai-pytorch
cd Vitis
sh quantize.sh
source compile.sh u50
```

## Ejecutar en Alveo U50
(TO DO)