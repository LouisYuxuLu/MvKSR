# <p align=center>  MvKSR: Multi-view Knowledge-guided Scene Recovery for Hazy and Rainy Degradation</p>

<div align="center">



</div>

---
>**MvKSR: Multi-view Knowledge-guided Scene Recovery for Hazy and Rainy Degradation**<br>  [Dong Yang](https://www.yangdong.info/), Wenyu Xu, [Yuan Gao](https://scholar.google.com.hk/citations?hl=zh-CN&user=4JpRnU4AAAAJ&view_op=list_works&sortby=pubdate), [Yuxu Lu<sup>*</sup>](https://scholar.google.com.hk/citations?user=XXge2_0AAAAJ&hl=zh-CN), Jingming Zhang, [Yu Guo](https://scholar.google.com/citations?user=klYz-acAAAAJ&hl=zh-CN)  (* indicates corresponding author) <br> 
>Under Review

> **Abstract:** *High-quality imaging is crucial for ensuring safety supervision and intelligent deployment in fields like transportation and industry. It enables precise and detailed monitoring of operations, facilitating timely detection of potential hazards and efficient management. However, adverse weather conditions, such as atmospheric haziness and precipitation, can have a significant impact on image quality. When the atmosphere contains dense haze or water droplets, the incident light scatters, leading to degraded captured images. This degradation is evident in the form of image blur and reduced contrast, increasing the likelihood of incorrect assessments and interpretations by intelligent imaging systems (IIS). To address the challenge of restoring degraded images in hazy and rainy conditions, this paper proposes a novel multi-view knowledge-guided scene recovery network (termed MvKSR). Specifically, guided filtering is performed on the degraded image to separate high/low-frequency components. Subsequently, an en-decoder-based multi-view feature coarse extraction module (MCE) is used to coarsely extract features from different views of the degraded image. The multi-view feature fine fusion module (MFF) will learn and infer the restoration of degraded images through mixed supervision under different views. Additionally, we suggest an atrous residual block to handle global restoration and local repair in hazy/rainy/mixed scenes. Extensive experimental results demonstrate that MvKSR outperforms other state-of-the-art methods in terms of efficiency and stability for restoring degraded scenarios in IIS. The source code is available at \url{https://github.com/LouisYuxuLu/MvKSR}.*
<hr />

## Requirement

- Python 3.7
- Pytorch 1.12.0


## Network Architecture
![Image](images/Network.jpg)

## Test
The pre-trained model will be provided after the paper is accpeted.

## Visual Results on Synthetic Images
![Image](images/Figure_Syn.jpg)

## Visual Results on Real-world Images
![Image](images/Figure_Real.jpg)

## Citation

```
Continue Update
