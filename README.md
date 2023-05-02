# Creating slow-motion Echocardiography
- The purpose of repositroy: sharing the codes for creating slow-motion echocardiography

As you know, ischemic heart disease remains a major cause of mortality worldwide, despite a growing body of evidence regarding the development of appropriate treatments and preventive measures. Echocardiography has a pivotal role to diagnose cardiovascular disease, evaluate severity, determine treatment options, and predict prognosis. 

Stress echocardiography is applied to evaluate the presence of myocardial ischemia or the severity of valvular disease using some drugs or during exercise. However, diagnostic accuracy depends on the physicians’ experience and image quality due to its high-rate image. We assume that the optimization of video frame rate with the same image quality might contribute to improved evaluation of echocardiography in a difficult setting including evaluation for patients with very fast heart rate. 

Reference:  **Super-SloMo** (https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
PyTorch implementation of "Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation" by Jiang H., Sun D., Jampani V., Yang M., Learned-Miller E. and Kautz J. [[Project]](https://people.cs.umass.edu/~hzjiang/projects/superslomo/) [[Paper]](https://arxiv.org/abs/1712.00080)

![SuperSlomo_NVIDIA_paper](https://user-images.githubusercontent.com/58348086/231926369-7d347036-fcd0-49e0-ab11-3eb6ce0e456a.png)


Acknowldgement:
The project members sincerely appreciate the previous work (Super-Slomo) by NVIDIA. 


Slow-motion Echocardiography:<br/>Arbitrary time interpolation using echocardiographic images
------------------------------------------------------------------------------
For details, see the accompanying paper.<br/>

 (Submitted as of May 2023）<br/>
 Development of artificial intelligence-based slow-motion echocardiography and clinical usefulness for evaluating regional wall motion abnormalities
> [**XXXXXX**](https://XXXXXXXXXXXX)　(PDF will be uploaded here after acceptance)
<br/>


Authors: <br/>
Yuki Sahashi MD*, MSc, Takeshita Ryo*, Takatomo Watanabe MD, PhD, Takuma Ishihara PhD, Ayako Sekine, Daichi Watanabe, Takeshi Ishihara MD, Hajime Ichiryu MD, Susumu Endo, Daisuke Fukuoka PhD, Takeshi Hara Ph, Hiroyuki Okura MD PhD, 

Dataset
-------
In this paper, all echocardiographic data were acquired using GE ultrasound equipment. 
We are very sorry that data are not shared for privacy purposes. 
<br/>

More than 1300 echocardiographic video data obtained from about 120 patients were trained into this paper.
The authors used a GPU (GeForce Titan, 24GB) for training and inference.
Please use your own dataset to create the arbitrary-time slow-motion echocardiography.

## Prerequisites
Model implementation and statistical analysis were performed using the Python (version 3.8) with Pytorch (version. 1.8.0) and R (version. 3.4.1), respectively. Model training was conducted using a graphics processing unit (GeForce Titan RTX 24 GB, NVIDIA, Santa Clara, California).

This project is implemented for Python 3, and depends on the following packages:
  - NumPy
  - PyTorch
  - Torchvision
  - sklearn
  - tqdm
  - tensorflow for tensorboard
  - etc.


Examples
--------
We show examples of our slow-motion echocardiography below.
![mp4 version](https://github.com/YukiSahashi/slowmotion_echocardiography/blob/main/docs/Project_supplemental_figure(2).mp4)

- Original (High heart rate due to exercise)

![original](https://user-images.githubusercontent.com/58348086/233840538-467026dc-2241-4bbb-bf3b-a291b2cdf67f.gif)

- 0.25x Slow motion echocardiography (same fps and 4x numbers of frames)

![0 25](https://user-images.githubusercontent.com/58348086/233840596-b3004af4-484d-4024-b456-f1ebed600244.gif)


- 0.125x Slow motion echocardiography (same fps and 8x numbers of frames)

![0 125](https://user-images.githubusercontent.com/58348086/233840623-93d5d172-0ce1-4e74-8553-f242507ace1b.gif)



One subject has normal cardiac function, another has a slight regional wall motion abnormalities.

 - example1 (Without Regional Wall motion abnormalities) <br/> 
   - video (Original Stress Echocardiography(Top)  <br/>
   - AI-based 0.25x Slow motion echocardiography (same fps and 4-times numbers of frames) (Left Bottom)  <br/> 
   - Manual 0.25x video (0.25 fps and same frame numbers as original)(Right bottom))
   
![video1](https://user-images.githubusercontent.com/58348086/233838418-a0cf7150-7b70-4ec4-be4c-3c285e1faea6.gif)

 - example2 (With Regional Wall motion abnormalities) <br/> 
   - video (Original Stress Echocardiography(Top)  <br/>
   - AI-based 0.25x Slow motion echocardiography (same fps and 4-times numbers of frames) (Left Bottom)  <br/> 
   - Manual 0.25x video (0.25 fps and same frame numbers as original)(Right bottom))    

![video2](https://user-images.githubusercontent.com/58348086/233838476-a69f2b30-77e6-4ee8-a68f-8670f1ba1f93.gif)


Usage Sample
-----
Note that the directory, image size should be modified based on each user's setting.
```bash
DICOM Dataset
inputs
└── dataset_folder
    ├── train_Dicoms
    │   ├── Dicom1 (xxx.dcm)
    │   ├── Dicom2 (yyy.dcm)
    │   ├── ...
    │
    └── test_Dicoms
       ├── Dicom1
       ├── Dicom2
       ├── ...

```

## Step 1　Dataset Preparation
```bash
python create_train_dataset.py 

--trainDicoms_folder path/to/folder/containig/dicoms/for/train # Dicom dataset for training
--testDicoms_folder path/to/folder/containig/dicoms/for/test # Dicom dataset for test
--dataset_folder path/to/output/dataset/folder 
--width resize image width #specify dataset image size :defalut 640
--height resize image height
```

```bash
python create_train_dataset.py --trainDicoms_folder dataset_folder/trainDicoms --testDicoms_folder dataset_folder/test_Dicoms --dataset_folder dataset_folder --width 640 --height 640
```

## Step 2　Training
```bash
inputs
└── dataset_folder
    ├── train_Dicoms
    ├── test_Dicoms
    ├── checkpoint 
    ├── log
    │
    ├── train 
    │   ├── 0 
    │   │   ├── frame00(000000.png)
    │   │   ├── frame01(000001.png)
    │   │   ├── ...
    │   │   └── frame12(000012.png)
    │   │
    │   ├── 1 
    │   │   ├── frame00
    │   │   ├── frame01
    │   │   ├── ...
    │   │   └── frame12
    │   │
    │   ├── ...
    │   │
    │   └── N
    │
    ├── test
    │   ├── 0
    │   │   ├── frame00
    │   │   ├── frame01
    │   │   ├── ...
    │   │   └── frame12
    │   │
    │   ├── 1 
    │   │   ├── frame00
    │   │   ├── frame01
    │   │   ├── ...
    │   │   └── frame12
    │   │
    │   ├── ...
    │   │
    │   └── N
    │
    └── validation
        ├── i 
        │   ├── frame00
        │   ├── frame01
        │   ├── ...
        │   └── frame12
        │
        ├── j
        │   ├── frame00
        │   ├── frame01
        │   ├── ...
        │   └── frame12
        │
        ├── ...
        │
        └── N
```

```bash
python train.py

--dataset_root path/to/dataset/folder/containing/train-test-validation/folders #Dataset path for train-test-validation dataset
--checkpoint_dir path/to/folder/for/saving/checkpoints 
--log_dir path/to/log/for/loss-valLoss-PSNR-SSIM/using/tensorboard/ 
--width image width #specify dataset image size :defalut 640
--height image height
```
```bash
python train.py --dataset_root dataset_folder --checkpoint_dir dataset_folder/checkpoint --log_dir dataset_folder/log --width 640 --height 640
```

## Step 3 Generating slowmotion echocardiography
```bash
inputs
└── dataset_folder
    ├── Video_folder
    │   ├── Video1
    │   ├── Video2
    │   ├── ...
    │
    └── DICOM_folder
       ├── DICOM1
       ├── DICOM2
       ├── ...
    
```

Please note:
sf indicates how many times the image is to be increased. (e.g. 2,4,8,12)

```bash
python video_to_slomo_SF.py
--ffmpeg_dir path/to/ffmpeg.exe #ffmpeg:https://ffmpeg.org/  Enter apps in this directory
--inputDir path/to/input/video or DICOM/folder 
--sf  #the number of increase in frame per second (ex: 4, 8, 12)
--outputDir path/to/output/folder #Path for output

```
#### (1) Video to slomo
```bash
python video_to_slomo.py --ffmpeg_dir path/to/ffmpeg --checkpoint dataset_folder/checkpoint/SuperSloMo39.ckpt --inputDir Video_folder --gpu 0 --sf 8 --outputDir path/to/output/folder --width 640 --height 640
```
#### (2) DICOM to slomo
```bash
python video_to_slomo.py --ffmpeg_dir path/to/ffmpeg --checkpoint dataset_folder/checkpoint/SuperSloMo39.ckpt --inputDir DICOM_folder --gpu 0 --sf 8 --outputDir path/to/output/folder --width 640 --height 640
```
