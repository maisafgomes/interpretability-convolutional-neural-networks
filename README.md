# Trabalho de Conclusão de Curso - Interpretabilidade em Redes Neurais Convolucionais

## Descrição

Este trabalho tem como objetivo utilizar atécnicas de interpretabilidade em uma rede neural convolucional para a detecção de glaucoma em
imagens de fundo de olho.

O modelo de rede neural escolhido foi o [ResNet](https://arxiv.org/abs/1512.03385)

As tecnicas de interpretabilidade foram a [Grad-CAM](https://doi.org/10.1371/journal.pone.0130140) e [LRP](https://doi.org/10.1007%2Fs11263-019-01228-7)

## Base de Dados

Para o treinamento e validação do modelo foram utilizadas duas bases publicas [Acrima](https://figshare.com/articles/dataset/CNNs_for_Automatic_Glaucoma_Assessment_using_Fundus_Images_An_Extensive_Validation/7613135) e [Rim-One](https://github.com/miag-ull/rim-one-dl). Além dessas, outra base foi criada fazendo a junção das imagens das duas bases publicas. 
