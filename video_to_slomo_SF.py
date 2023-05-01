#!/usr/bin/env python3
import argparse
import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import model
import dataloader
import platform
from tqdm import tqdm

from pathlib import Path

import numpy as np
import pydicom
import io
from pydicom.encaps import encapsulate
from natsort import natsorted
import time
import cv2

import uuid
import subprocess

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, default="", help='path to ffmpeg.exe') # ffmpeg.exeが保存されているフォルダのパス
parser.add_argument("--checkpoint", type=str, required=True, help='path of checkpoint for pretrained model') #学習時に保存したパラメータに関するファイル(.ckpt)のパス　
                                                                                                            #ex)~/SuperSloMo{}.ckpt ※{}には、数字が入る。保存されている中で最も大きい番号のファイルの使用を推奨。
parser.add_argument("--inputDir", type=str, required=True, help='path of Dicom')
parser.add_argument("--gpu", type=int, default=0, help='device number. Default: 0')
parser.add_argument("--sf", type=int, required=True, help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
parser.add_argument("--fps_magnification", type=int, default=1, help='frame rate magnification of slomo video')
parser.add_argument("--batch_size", type=int, default=1, help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument("-d","--dicom_flag", action='store_true', help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument("--outputDir", type=str, required=True, help='Specify output file name.')
parser.add_argument("--width", type=int, required=True, help='width of input image for train')
parser.add_argument("--height", type=int, required=True, help='height of input image for train')

parser.add_argument("--Red", type=float, default=0.186, help='Training Channel Wise Mean Red')
parser.add_argument("--Green", type=float, default=0.184, help='Training Channel Wise Mean Green')
parser.add_argument("--Blue", type=float, default=0.206, help='Training Channel Wise Mean Blue')
args = parser.parse_args()


def check():
    error = ""
    #if (args.sf < 2):
    #    error = "Error: --sf/slomo factor has to be atleast 2"
    if (args.batch_size < 1):
        error = "Error: --batch_size has to be atleast 1"
    #if (args.fps < 1):
    #    error = "Error: --fps has to be atleast 1"
    #if ".mkv" not in args.output:
    #    error = "output needs to have mkv container"
    return error

#DICOMファイルをPNG形式に展開する
def dicomToImage(srcPath,destPath):
    ds=pydicom.dcmread(srcPath)
    fps = ds[0x0018,0x0040].value
    ori_h = ds[0x0028,0x0010].value
    ori_w = ds[0x0028,0x0011].value
    h = args.height
    w = args.width
    top = 0
    left = 0
    if ori_h != h or ori_w != w:
        top = int(ori_h/2 - h/2)
        left = int(ori_w/2 - w/2)
        if left < 0 or top < 0:
            return -1
    print(ds)
    imgYBR=ds.pixel_array
    # assert ds[0x0028,0x0008].value == imgYBR.shape[0]
    for i in range(imgYBR.shape[0]):
        imgRGB=pydicom.pixel_data_handlers.convert_color_space(imgYBR[i,:,:,],'YBR_FULL_422','RGB')
        pil_img = Image.fromarray(imgRGB)
        crop_img = pil_img.crop((left, top, left+w, top+h))
        destFile = os.path.join(destPath,'{:06}.png'.format(i))
        crop_img.save(destFile)

    if imgYBR.shape[0] == 0:
        return -2
    else:
        return fps

#動画ファイルをPNG形式に展開する
def videoToImage(srcPath,destPath):
    cap = cv2.VideoCapture(srcPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ori_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ori_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = args.width
    h = args.height
    top = 0
    left = 0
    if ori_w != w or ori_h != h:
        left = int(ori_w/2 - w/2)
        top = int(ori_h/2 -h/2)
        if left < 0 or top < 0:
            return -1
    n = 0
    while True:
        ret, frame = cap.read()
        if ret:
            crop_frame = frame[top:top+h, left:left+w]
            cv2.imwrite(os.path.join(destPath,'{:06}.png'.format(n)),crop_frame)
            n += 1
        else:
            return fps
   

#MKVファイル形式で保存する
def create_video(dir,destPath,fps):
    #IS_WINDOWS = 'Windows' == platform.system()

    #if IS_WINDOWS:
    #    ffmpeg_path = os.path.join(args.ffmpeg_dir, "ffmpeg")
    #else:
    #    ffmpeg_path = "ffmpeg"

    ffmpeg_path = os.path.join(args.ffmpeg_dir, "ffmpeg")
    error = ""
    #print('{} -y -r {} -i {}/%d.png -vcodec ffvhuff {}'.format(ffmpeg_path, fps, dir, destPath))
    retn = os.system('{} -y -r {} -i {}/%6d.png -vcodec ffvhuff "{}"'.format(ffmpeg_path, fps, dir, destPath))
    if retn:
        error = "Error creating output video. Exiting."
    return error



def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')


#DICOMファイルに保存する部分
def ensure_even(stream):
    # Very important for some viewers
    if len(stream) % 2:
        return stream + b"\x00"
    return stream


def main(inputDir,slomoFactor,fps_mag,destDir,gpu,dflag):
    # Check if arguments are okay
    error = check()
    if error:
        print(error)
        exit(1)
    
    # ＧＰＵの選択
    if gpu==0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if gpu==1:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Channel wise mean calculated on adobe240-fps training dataset
    #mean = [0.429, 0.431, 0.397] Adobe240

    mean = [args.Red, args.Green, args.Blue]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,std=std)

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
    # - Removed per channel mean subtraction for CPU.
    if (device == "cpu"):
        transform = transforms.Compose([transforms.ToTensor()])
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        TP = transforms.Compose([revNormalize, transforms.ToPILImage()])
    
    input_list = sorted(os.listdir(inputDir))
    for input in input_list:
        # get fileName 
        if dflag:
            fileName = os.path.basename(input)
        else:
            fileName = os.path.basename(input).split('.')[0]
        
        inputPath = os.path.join(inputDir,input)
        # 出力フォルダ作成 (一時的なフォルダ)   
        imgPath = os.path.join(destDir, fileName)
        if not os.path.isdir(imgPath):
            os.mkdir(imgPath)
        outputPath = os.path.join(destDir, '{}_Slomo_x{}'.format(fileName, slomoFactor))
        if not os.path.isdir(outputPath):
            os.mkdir(outputPath)
        
        # Load data
        if dflag:
            fps = dicomToImage(inputPath,imgPath)
            if fps == -1:
                print("ERROR: Image size is too little !!!")
                exit(1)
            elif fps == -2:
                print("ERROR: DICOM image data is Nothing !!!")
                exit(1)
        else:
            fps = videoToImage(inputPath,imgPath)
            if fps == -1:
                print("ERROR: Image size is too little !!!")
                exit(1)

        destPathMKV=os.path.join(destDir,'{}_Slomo_x{}_fps{}.mkv'.format(fileName, slomoFactor, int(fps*fps_mag)))

        
        videoFrames = dataloader.Video(root=imgPath, transform=transform)
        lenFrames = videoFrames.__len__()
        lenMax = lenFrames*slomoFactor
        print('Number of Frames: {}->{}'.format(lenFrames+1,lenMax+1))
        videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=args.batch_size, shuffle=False)

        # Initialize model
        flowComp = model.UNet(6, 4)
        flowComp.to(device)
        for param in flowComp.parameters():
            param.requires_grad = False
        ArbTimeFlowIntrp = model.UNet(20, 5)
        ArbTimeFlowIntrp.to(device)
        for param in ArbTimeFlowIntrp.parameters():
            param.requires_grad = False

        flowBackWarp = model.backWarp(videoFrames.dim[0], videoFrames.dim[1], device)
        flowBackWarp = flowBackWarp.to(device)

        dict1 = torch.load(args.checkpoint, map_location='cpu')
        ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
        flowComp.load_state_dict(dict1['state_dictFC'])

        # Interpolate frames
        frameCounter = 0

        with torch.no_grad():
            for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):
                I0 = frame0.to(device)
                I1 = frame1.to(device)

                flowOut = flowComp(torch.cat((I0, I1), dim=1))
                F_0_1 = flowOut[:,:2,:,:]
                F_1_0 = flowOut[:,2:,:,:]

                # Save reference frames in output folder
                for batchIndex in range(args.batch_size):
                    (TP(frame0[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, '{:0>6}.png'.format(frameCounter + slomoFactor * batchIndex)))
                
                frameCounter += 1

                # Generate intermediate frames
                for intermediateIndex in range(1, slomoFactor):
                    t = float(intermediateIndex) / slomoFactor
                    temp = -t * (1 - t)
                    fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                    F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                    F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                    g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                    g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

                    intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                    F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                    F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                    V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                    V_t_1   = 1 - V_t_0

                    g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                    g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

                    wCoeff = [1 - t, t]

                    Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                    # Save intermediate frame
                    for batchIndex in range(args.batch_size):
                        (TP(Ft_p[batchIndex].cpu().detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, '{:0>6}.png'.format(frameCounter + slomoFactor * batchIndex)))
                    frameCounter += 1

                # Set counter accounting for batching of frames
                frameCounter += slomoFactor * (args.batch_size - 1)

                if frameCounter >= lenMax:
                    for batchIndex in range(args.batch_size):
                        (TP(frame1[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, '{:0>6}.png'.format(frameCounter + slomoFactor * batchIndex)))


        # Generate slomo video
        create_video(outputPath, destPathMKV ,fps*fps_mag)
        # 画像保存フォルダの削除
        rmtree(imgPath)
        rmtree(outputPath)

main(args.inputDir, args.sf, args.fps_magnification, args.outputDir, args.gpu, args.dicom_flag)
    
exit(0)

