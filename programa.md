# Guía

## Semana 16-23 Nov

1. Organizar el github: archivos y libreta entrenamiento DONE
2. Chequear datos entrenamiento: standarización DONE
3. Montar encoder-decoder	  DONE
    4. https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e
    5. https://vaibhaw-vipul.medium.com/building-autoencoder-in-pytorch-34052d1d280c 
5. Matriz soporte vitis DONE
6. 3 papers stereo matching (Juan) LACK
7. (Opcional) Montar red same resolution LACK
8. Fix docker con gpu - https://github.com/NVIDIA/nvidia-docker/issues/1243 - docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]. LACK

## Semana 19-26 Nov

1. Notebook visualización: visualizar output batch dataloader. Montar gráfico ejemplo input/output
2. Entrenar disparity: (inputL, inputR) -> Disparity
    2.1 img = torch.cat([imgL, imgR], dim=1) Cambiar canales input autoencoder a 6 y el putput a 1. DONE
4. Tuto: conversión TOrchScript JIT del pth https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
    4.1 Chequear VITIS: convertir el grafo (independiente o no de 1.) Obtener XModel
5. Comenzar intro: descripción problema (stereo mattching -> disparity) basándose en refs.

## Semana 25 Nov - 3 Dic

1. Learning rate 5e-5, sin weight decay DONE 
2. Chequear pipeline datos inoput/output:
    + con MobileStereoNet
    + Interpolacion ground truth
5.  (Opcional) Montar tensorboard logging training 
    pip install tensorboard; tensorboard --logdir <folder de los logs> https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
4. (Importante) MIrar metrica en MobileStereoNet (EPE, D1) e implementarla en script y en tensorboard
5. Mir resnet comoi encoder DONE
6. (IMportante)  Probar de bajar resnet o baseline  a VITIS DONE

## Semana 1 Dic - 3 Dic
    
1. Entrenar Sceneflow con Backbone
2. Mirar exactamente qué hace exactamente MobileStereoNet a nivel de código
    + Interface cost volume
    + Disparity Regression
    + Tamaños?
    + MaxDisp
    + Mascara en input
    + Loss
3. Crear archivo dde entorno con conda o pip para reproducción 
    
# Tutos

1. https://pytorch.org/tutorials
    1.1 Learning Pytorch
    1.2 Load and save models

# References

1. Shamsafar, Faranak, et al. "MobileStereoNet: Towards Lightweight Deep Networks for Stereo Matching." arXiv preprint arXiv:2108.09770 (2021). https://arxiv.org/pdf/2108.09770.pdf
2. https://proceedings.neurips.cc/paper/2020/file/fc146be0b230d7e0a92e66a6114b840d-Paper.pdf
