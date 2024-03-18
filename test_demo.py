# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:48:05 2020

@author: Administrator
"""


import torch
import torch.nn as nn

#from tensorboardX import SummaryWriter
import numpy as np
import cv2
import time
import os
from MvKSR import *
import utils_train


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_checkpoint(checkpoint_dir,IsGPU):
    
	if IsGPU == 0:
		model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
		net = MvKSR()
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).cuda()
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(model_info['optimizer'])
		cur_epoch = model_info['epoch']
	else:

		model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar',map_location=torch.device('cpu'))
		net = MvKSR()
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids)
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(model_info['optimizer'])
		cur_epoch = model_info['epoch']


	return model, optimizer,cur_epoch

def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
			print( param_group['lr'])
	return optimizer

def train_psnr(train_in,train_out):
	
	psnr = utils_train.batch_psnr(train_in,train_out,1.)
	return psnr


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])

def guideFilterTest(img, winSizes=[5,13,25], eps=0.1, target = False):
    
    HFImg = np.zeros((img.shape))
    LFImg = np.zeros((img.shape))
    
    I = 0.333 * img[0,:,:] + 0.333 * img[1,:,:] + 0.333 * img[2,:,:]

    
    mean_I_5 = cv2.blur(I, (winSizes[0],winSizes[0]))
    mean_II_5 = cv2.blur(I*I, (winSizes[0],winSizes[0]))
    var_I_5 = mean_II_5 - mean_I_5 * mean_I_5
    a_5 = var_I_5 / (var_I_5 + eps)
    b_5 = mean_I_5 - a_5*mean_I_5
    mean_a_5 = cv2.blur(a_5, (winSizes[0],winSizes[0]))
    mean_b_5 = cv2.blur(b_5, (winSizes[0],winSizes[0]))
    LFImg_5 = mean_a_5*I + mean_b_5
    HFImg_5 = I - LFImg_5
    
    
    mean_I_13 = cv2.blur(I, (winSizes[1],winSizes[1]))
    mean_II_13 = cv2.blur(I*I, (winSizes[1],winSizes[1]))
    var_I_13 = mean_II_13 - mean_I_13 * mean_I_13
    a_13 = var_I_13 / (var_I_13 + eps)
    b_13 = mean_I_13 - a_13*mean_I_13
    mean_a_13 = cv2.blur(a_13, (winSizes[1],winSizes[1]))
    mean_b_13 = cv2.blur(b_13, (winSizes[1],winSizes[1]))
    LFImg_13 = mean_a_13*I + mean_b_13
    HFImg_13 = I - LFImg_13    


    mean_I_25 = cv2.blur(I, (winSizes[2],winSizes[2]))
    mean_II_25 = cv2.blur(I*I, (winSizes[2],winSizes[2]))
    var_I_25 = mean_II_25 - mean_I_25 * mean_I_25
    a_25 = var_I_25 / (var_I_25 + eps)
    b_25 = mean_I_25 - a_25*mean_I_25
    mean_a_25 = cv2.blur(a_25, (winSizes[2],winSizes[2]))
    mean_b_25 = cv2.blur(b_25, (winSizes[2],winSizes[2]))
    LFImg_25 = mean_a_25*I + mean_b_25
    HFImg_25 = I - LFImg_25

    HFImg[0,:,:] = HFImg_5
    LFImg[0,:,:] = LFImg_5
    HFImg[1,:,:] = HFImg_13
    LFImg[1,:,:] = LFImg_13
    HFImg[2,:,:] = HFImg_25
    LFImg[2,:,:] = LFImg_25
    
    if target is False:
        return (HFImg ,LFImg)
    else:
        return (HFImg_5 , LFImg_5)

def FastguideFilter(I, p, winSize, eps=0.05, s=0.5):
    
    I = 0.333 * img[0,:,:] + 0.333 * img[1,:,:] + 0.333 * img[2,:,:]
    p = I.copy()
    
    h, w = I.shape[:2]
    
    #缩小图像
    size = (int(round(w*s)), int(round(h*s)))
    
    small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    small_p = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    
    #缩小滑动窗口
    X = winSize[0]
    small_winSize = (int(round(X*s)), int(round(X*s)))
    
    #I的均值平滑
    mean_small_I = cv2.blur(small_I, small_winSize)
    
    #p的均值平滑
    mean_small_p = cv2.blur(small_p, small_winSize)
    
    #I*I和I*p的均值平滑
    mean_small_II = cv2.blur(small_I*small_I, small_winSize)
    
    mean_small_Ip = cv2.blur(small_I*small_p, small_winSize)
    
    #方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I #方差公式
    
    #协方差
    cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p
   
    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a*mean_small_I
    
    #对a、b进行均值平滑
    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)
    
    #放大
    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)
    
    q = mean_a*I + mean_b
    
    return q

if __name__ == '__main__': 	
	checkpoint_dir = './checkpoint/'
	test_dir = './input'
	result_dir = './output'    
	testfiles = os.listdir(test_dir)
    
	IsGPU = 0    #GPU is 1, CPU is 0

	print('> Loading dataset ...')

	lr_update_freq = 30
	model,optimizer,cur_epoch = load_checkpoint(checkpoint_dir,IsGPU)          
	  
	if IsGPU == 0:				
		for f in range(len(testfiles)):
			model.eval()
			with torch.no_grad():
				img = cv2.imread(test_dir + '/' + testfiles[f])
				imgg = cv2.imread(test_dir + '/' + testfiles[f],0)/255
				h,w,c = img.shape
				img_ccc = img / 255.0
				img_h = hwc_to_chw(img_ccc)
				input_var = torch.from_numpy(img_h.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
              
                
				guideh,guidel = guideFilterTest(img_h, target = False)
              
				input_varh = torch.from_numpy(guideh.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
				input_varl = torch.from_numpy(guidel.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()   
				s = time.time()
				h_out,l_out,i_out,e_out = model(input_var,input_varh,input_varl)     
				e = time.time()   
  
				print(e-s)  
	             	
				e_out = chw_to_hwc(e_out.squeeze().cpu().detach().numpy())	                  
 
		
				cv2.imwrite(result_dir + '/' + testfiles[f][:-4] +'.jpg',e_out*255)				

                
