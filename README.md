# Synthetic X-Ray STS Research 2020 - 2021

Reconstructing CT volumes in 3D using convolutional encoder-decoder representation frameworks and generative adversarial networks. Initial results can be accessed in the 'IPYNB' folder via Google Colaboratory or JupyterLab. Training is done on a single Tesla V100 GPU hosted on Amazon Web Services (AWS). 


Single View Autoencoder Results:
MSE: 0.0011900706052539642 (avg)
Lung DICE: 0.9411571828756816 (avg)
SSIM: 0.8068173427439007 (avg)

![Single Result](/Images/single_view.jpg)

Double View Autoencoder Results:
MSE: 0.000952999815192659 (avg)
Lung DICE: 0.971042315 (avg)
SSIM: 0.83141169669 (avg)

![Double Result](/Images/double_view.jpg)