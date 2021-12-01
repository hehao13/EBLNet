
## Here we give Three different glass-like object segmentation datasets example, including Trans10k, BDD, and MSD.

## Trans10k

First of all, please download the dataset from [Tran10K website](https://xieenze.github.io/projects/TransLAB/TransLAB.html).
Then put the data under 'PATH/TO/YOUR/DATASETS/Trans10k'. Data structure is shown below.
```
Trans10k/
├── test
│   ├── images
│   └── masks
├── test_easy
│   ├── images
│   └── masks
├── test_hard
│   ├── images
│   └── masks
├── train
│   ├── images
│   └── masks
└── validation
    ├── images
    └── masks
```
 Note that, we merge the validation easy and validation hard.
 For the test set, we keep the original split of easy and hard, at the same time, we also merge both of them to test file.


## GDD

First, please request and download the dataset from [GDD_website](https://mhaiyang.github.io/CVPR2020_GDNet/index). 
Then put the data under 'PATH/TO/YOUR/DATASETS/GDD'. Data structure is shown below.
```
GDD/
├── test
│   ├── images
│   └── masks
└── train
    ├── images
    └── masks
```

## MSD

First, please download the dataset from [MSD_website](https://mhaiyang.github.io/ICCV2019_MirrorNet/index.html).
Then put the data under 'PATH/TO/YOUR/DATASETS/MSD'. Data structure is shown below.
```
MSD/
├── test
│   ├── images
│   └── masks
└── train
    ├── images
    └── masks
```
After that, you can either change the `config.py` or do the soft link according to the default path in config.

For example, 

Suppose you soft line all your datasets at `./data`, then update the dataset path in `config.py`,
```
mkdir data
ln -s PATH/TO/YOUR/DATASETS/* PATH/TO/YOUR/CODE/data
# MSD Dataset Dir Location
__C.DATASET.MSD_DIR = './data/MSD'
# GDD Dataset Dir Location
__C.DATASET.GDD_DIR = './data/GDD'
# Trans10k Dataset Dir Location
__C.DATASET.TRANS10K_DIR = './data/Trans10k'
``` 
