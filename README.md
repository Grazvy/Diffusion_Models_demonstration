### A Denoising Diffusion Probabilistic Model

introduction...

reference paper and slides

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

