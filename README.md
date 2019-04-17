# Alpha_GAN
A Tensorflow implementation to reproduce the results presented in Alpha GAN paper. In this implementation, I have tried alpha-gan model on two real world datasets, namely; Cifar10 and celebA. The paper performs experiments on inception score for quantitative results. However, we restrict only to the qualitative analysis as the famous Inception score has been empirically shown with suboptimalities by Barratt et al. For a detailed read, follow [this](https://arxiv.org/abs/1801.01973).

## Setup
* Python 3.5+
* Tensorflow 1.9

## Relevant Code Files

File config.py contains the hyper-parameters for Alpha_gan reported results.

File alpha_gan.py contains the code to train Alpha_gan model.

Similarly, as the name suggests, file alpha_gan_inference.py contains the code to test the trained Alpha_gan model.

## Usage
### Training a model
NOTE: For celebA, make sure you have the downloaded dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and keep it in the current directory of project.
```
python alpha_gan.py
```
### Test a trained model 
 
First place the model weights in a directory whose name is mentioned in a variable named model_directory (refer to alpha_gan_inference.py) and then:
```
python alpha_gan_inference.py 
```
## Emprical Observations

The model is notoriously hard to train. I found the hyper-paramerters mentioned in the paper to be vague as only the hyper-parameter spreads are mentioned but it is hard to know which parameters were finally chosen to reproduce the results reported in the paper.

However, one may note that the alpha-gan is 4 tier architecture comprising of encoder, decoder(authors call it generator), discriminator and code-discriminator and compared to VAE-GAN results, I qualitatively observe no significant gain.

For code-discriminator, please **avoid use of batchnorm layers**. We spend a couple of days due to this. If you make it work, do message me over your github repository.

For encoder network, use the RELU activations for intermediate layers. Although in general, we are free to choose any activation function for encoder but in the alpha_gan approach it act as a generator fooling the code-discriminator. Now, as according to DCGAN architecture guidelines, the generator should use RELU activations, therefore, our encoder is RELU activated.

I tried multiple schedules by re-arranging the updates like first updating the discriminator and code-discriminator followed by encoder and generator -- but I could not find any performance gains. 

For both the datasets it seems that alpha_gan focuses more on generations as compared to reconstruction ability. Also, the official paper reports reconstructions results only for the **training** data points. -- *I wonder why..?*

## Model Weights
[CelebA model weights](https://drive.google.com/drive/folders/1PMCN8DQsbWh6q-LNZihcflPvu_I62tnW?usp=sharing)

## Generations

Cifar10            |  Celeb-A
:-------------------------:|:-------------------------: 
![](https://github.com/PrateekMunjal/Alpha_GAN/blob/master/cifar10/generations-cifar.gif)  |  ![](https://github.com/PrateekMunjal/Alpha_GAN/blob/master/celebA/generations-celeba.gif)

## Reconstructions
*Qualitative analysis for CelebA dataset*

CelebA Original            |  CelebA Reconstruction
:-------------------------:|:-------------------------: 
![](https://github.com/PrateekMunjal/Alpha_GAN/blob/master/celebA/op/orig-img-10.png)  |  ![](https://github.com/PrateekMunjal/Alpha_GAN/blob/master/celebA/op/recons-img-10.png)
  
*Qualitative analysis for Cifar10 dataset*

Cifar10 Original            |  Cifar10 Reconstruction
  :-------------------------:|:-------------------------: 
  ![](https://github.com/PrateekMunjal/Alpha_GAN/blob/master/cifar10/op/orig-img-20.png)  |  ![](https://github.com/PrateekMunjal/Alpha_GAN/blob/master/cifar10/op/recons-img-20.png)
