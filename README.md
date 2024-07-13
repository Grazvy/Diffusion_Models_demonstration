### A Denoising Diffusion Probabilistic Model

Diffusion models are the new state-of-the-art family of deep generative models, being used popular AI systems, 
such as Dall-E 2 and Midjourney.<br/>
I wrote this [review paper](resources/Diffusion_Models.pdf) about diffusion models, based on relevant papers. My teaching goal is to provide
an overview and go down a specific path, until its application. Since this is what I wished for, when learning
about diffusion models, but none of the papers was providing. <br/>
So this repository is the practical summary of the theory discussed and an opportunity to play around with image 
synthesis. That's why most of the infrastructure, which is not relevant for diffusion models specifically, was hidden 
in the "utils" folder. <br/>
This was my part in the seminar "Data Mining" at TUM, which I am happy to share now. You can also check out the corresponding [presentation slides](https://docs.google.com/presentation/d/e/2PACX-1vRCoAqSb1gb5Khuh7aI0a_MwUcAwFF5lDqWjUEkTloc8UKY89TXRbYoVdEVcVz5u0XX9msbiLUEGdPM/pub?start=true&loop=false&delayms=3000).
The review paper explains the context, derives implemented algorithms, and also walks you through the Jupyter Notebook, 
explaining relevant methods.


### first steps
set up environment using miniconda: <br/>
(python 3.10)
```
conda install pytorch::pytorch
conda install pytorch::torchvision
conda install conda-forge::torchmetrics
conda install conda-forge::matplotlib-base
conda install conda-forge::tqdm
conda install fastai::opencv-python-headless
```

### download additional data (optional)

If you want to train on images of flowers, you can download this dataset: <br/>
https://www.kaggle.com/datasets/alxmamaev/flowers-recognition/data <br/>
Then add all files in the root folder as:<br/>
.<br/>
├── ...<br/>
├── flowers               
│   ├── daisy <br/>
│   ├── dandelion   
│   ├── rose <br/>
│   ├── sunflower <br/>
│   └── tulip              
└── ...

You cal also add your own data, with corresponding adjustment on the dataloader.


### CREDITS
This repo initially started with code from https://github.com/spmallick/learnopencv/tree/master/Guide-to-training-DDPMs-from-Scratch.

