# Synthetic X-Ray STS Research 2020 - 2021

Reconstructing CT volumes in 3D using convolutional encoder-decoder representation frameworks and generative adversarial networks. Initial results can be accessed in the 'IPYNB' folder via Google Colaboratory or JupyterLab. Training is done on a single Tesla V100 GPU hosted on Amazon Web Services (AWS). 

CT -> X-ray from:

A. Moturu and A. Chang, “Creation of synthetic x-rays to train a neural network to detect lung cancer.” http://www.cs.toronto.edu/pub/reports/na/Project_Report_Moturu_Chang_1.pdf, 2018.

X-ray -> CT from GAN (not made yet)

Single View Results:
MSE: 0.0011900706052539642 (avg)
Lung DICE: 0.9611571828756816 (avg)
SSIM: 0.8068173427439007 (avg)
