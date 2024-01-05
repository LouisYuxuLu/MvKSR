# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:14:37 2021

@author: Administrator
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize

class SmartVITS(nn.Module):
	def __init__(self):
		super(SmartVITS,self).__init__()

		self.mns = MainNetworkStructure(3,16)
         
	def forward(self,x,x_h,x_l):
        
		FIout, Hout, Lout, Iout = self.mns(x,x_h,x_l)
      
		return FIout, Hout, Lout, Iout


class MainNetworkStructure(nn.Module):
	def __init__(self,inchannel,channel):
		super(MainNetworkStructure,self).__init__()

		self.conv_in = nn.Conv2d(3,channel,kernel_size=1,stride=1,padding=0,bias=False)   
        
		self.bb_1 = ConvL(channel,channel)
		self.bb_2 = ConvL(channel,channel) 
		self.bb_3 = ConvL(channel,channel)         
		self.conv_out = nn.Conv2d(channel,3,kernel_size=3,stride=1,padding=1,bias=False)   
        
		self.fhlfgm = FHLFGM(channel)
		self.bhlfgm = BHLFGM(channel)
        
		self.ed = En_Decoder(9,channel) 
        
	def forward(self,x_i,x_h,x_l):
        
		x_ed = self.ed(x_i, x_h, x_l)
		x_hout, x_lout, x_iout, x_hr2, x_lr2, x_ir2 = self.fhlfgm(x_ed)
		x_bhr2, x_blr2, x_bir2                      = self.bhlfgm(x_ed,x_hout, x_lout, x_iout, x_hr2, x_lr2, x_ir2)        

		x_out =  self.conv_out(self.bb_3(self.bb_2(self.bb_1(x_bhr2 + x_blr2) + x_bir2)))
            
		return x_hout, x_lout, x_iout, x_out

    
class MRB(nn.Module):    #Mixed Residual Block (MRB)
	def __init__(self,channel,rate):                                
		super(MRB,self).__init__()

		self.conv_1_1 = nn.Conv2d(channel,int(0.5*channel),kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_1_2 = nn.Conv2d(channel,int(0.5*channel),kernel_size=3,stride=1,padding=rate,dilation=rate,bias=False)
		self.conv_2_1 = nn.Conv2d(channel,int(0.5*channel),kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2_2 = nn.Conv2d(channel,int(0.5*channel),kernel_size=3,stride=1,padding=rate,dilation=rate,bias=False)        
		self.conv_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_out = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.act = nn.PReLU(channel)

		self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
   
	def forward(self,x):
        
		x_1 = self.act(self.norm(torch.cat((self.conv_1_1(x),self.conv_1_2(x)),1)))
		x_2 = self.act(self.norm(torch.cat((self.conv_2_1(x_1),self.conv_2_2(x_1)),1)))
		x_out = self.act(self.norm(self.conv_out(x_2)) + x)

		return	x_out

    
class FHLFGM(nn.Module):    #Front High and Low Frequency Information Guidance Module
	def __init__(self,channel):                                
		super(FHLFGM,self).__init__()

		self.hr1 = MRB(channel,rate=6)
		self.hr2 = MRB(channel,rate=6)
		self.hr3 = MRB(channel,rate=6)
        
		self.lr1 = MRB(channel,rate=6)
		self.lr2 = MRB(channel,rate=6)
		self.lr3 = MRB(channel,rate=6)
        
		self.ir1 = MRB(channel,rate=6)
		self.ir2 = MRB(channel,rate=6)  
		self.ir3 = MRB(channel,rate=6)
        
		self.convI_out = nn.Conv2d(channel,1,kernel_size=3,stride=1,padding=1,bias=False)   
		self.convH_out = nn.Conv2d(channel,1,kernel_size=3,stride=1,padding=1,bias=False)
		self.convL_out = nn.Conv2d(channel,1,kernel_size=3,stride=1,padding=1,bias=False)        
            
	def forward(self,x_ihl):

		x_hr1 = self.hr1(x_ihl)
		x_hr2 = self.hr2(x_hr1)
		x_hr3 = self.hr3(x_hr2)
        
		x_lr1 = self.lr1(x_ihl)
		x_lr2 = self.lr2(x_lr1)
		x_lr3 = self.lr3(x_lr2)
        
		x_ir1 = self.ir1(x_ihl + x_hr1 + x_lr1)
		x_ir2 = self.ir2(x_ir1 + x_hr2 + x_lr2)
		x_ir3 = self.ir3(x_ir2 + x_hr3 + x_lr3)
                    
		x_hout = self.convH_out(x_hr3)
		x_lout = self.convL_out(x_lr3)
		x_iout = self.convI_out(x_ir3)
        
		return	x_hout, x_lout, x_iout, x_hr3, x_lr3, x_ir3
    

class BHLFGM(nn.Module):    #Back High and Low Frequency Information Guidance Module
	def __init__(self,channel):                                
		super(BHLFGM,self).__init__()

		self.hr1 = MRB(channel,rate=3)
		self.hr2 = MRB(channel,rate=3)
		self.hr3 = MRB(channel,rate=3)     
        
		self.lr1 = MRB(channel,rate=3)
		self.lr2 = MRB(channel,rate=3)
		self.lr3 = MRB(channel,rate=3)
        
		self.ir1 = MRB(channel,rate=3)
		self.ir2 = MRB(channel,rate=3)
		self.ir3 = MRB(channel,rate=3)
        
		self.convI_in = nn.Conv2d(1,channel,kernel_size=3,stride=1,padding=1,bias=False)   
		self.convH_in = nn.Conv2d(1,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.convL_in = nn.Conv2d(1,channel,kernel_size=3,stride=1,padding=1,bias=False)     
        
		self.convF_out = nn.Conv2d(channel,3,kernel_size=3,stride=1,padding=1,bias=False)   
      
            
	def forward(self, xed, x_fhout, x_flout, x_fiout, x_fhr2, x_flr2, x_fir2):

		x_bhr1 = self.hr1(self.convH_in(x_fhout) + x_fhr2)
		x_bhr2 = self.hr2(x_bhr1)
		x_bhr3 = self.hr2(x_bhr2)
        
		x_blr1 = self.lr1(self.convL_in(x_flout) + x_flr2)
		x_blr2 = self.lr2(x_blr1)
		x_blr3 = self.lr2(x_blr2)

		x_bir1 = self.ir1(self.convI_in(x_fiout) + xed + x_fir2 + x_bhr1 + x_blr1)
		x_bir2 = self.ir2(x_bir1 + x_bhr2 + x_blr2)           
		x_bir3 = self.ir2(x_bir2 + x_bhr3 + x_blr3)     
        
		return	x_bhr3, x_blr3, x_bir3
    
    
class ConvL(nn.Module):
	def __init__(self,inchannel,channel,norm=False):                                
		super(ConvL,self).__init__()

		self.conv = nn.Conv2d(inchannel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.act = nn.PReLU(channel)
		self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)
   
	def forward(self,x):
        
		x_out = self.act(self.norm(self.conv(x)))

		return	x_out      


class En_Decoder(nn.Module):
	def __init__(self,inchannel,channel):
		super(En_Decoder,self).__init__()
        
		self.el  = MRB(channel,rate=12)
		self.em  = MRB(channel*2,rate=6)
		self.es  = MRB(channel*4,rate=3)     
		self.ds  = MRB(channel*4,rate=3)
		self.dm  = MRB(channel*2,rate=6)
		self.dl  = MRB(channel,rate=12)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
		self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)   
         
		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)  

		self.conv_in = nn.Conv2d(inchannel,channel,kernel_size=3,stride=1,padding=1,bias=False)        
		self.conv_out = nn.Conv2d(channel,3,kernel_size=3,stride=1,padding=1,bias=False)    
    
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')

	def forward(self,x1,x2,x3):
             
		x_elin = self.conv_in(torch.cat((x1,x2,x3),1))
        
		elout  = self.el(x_elin)        
		emout  = self.em(self.conv_eltem(self.maxpool(elout)))        
		esout  = self.es(self.conv_emtes(self.maxpool(emout)))
        
		dsout  = self.ds(esout)
		dmout  = self.dm(self._upsample(self.conv_dstdm(dsout),emout) + emout)
		dlout  = self.dl(self._upsample(self.conv_dmtdl(dmout),elout) + elout)

		return dlout

    