# 💧 WMSuite: Image Watermark Processing Toolkit

WMSuite is a lightweight Python toolkit that integrates several existing neural-network–based invisible watermarking algorithms.  
The goal of this project is not to propose new watermarking methods, but to provide a **simple and unified interface** for:

- embedding watermarks into images from a specified folder, and  
- extracting watermarks from images in a specified folder.

Currently, WMSuite includes support for 6–7 publicly available watermarking models, with a consistent configuration and execution workflow to make experimentation easier.


## 🚀 Quick Start Guide

### 1. Environment Setup

To set up the project environment, follow the steps below.
```bash
conda create -n <env_name> python=3.10
```
Activate the newly created environment.
```bash
conda activate <env_name>
```
Install all required Python dependencies from the provided requirements file.
```
pip install -r requirements.txt
```


### 2. Model Weights Configuration 📥

Several neural-network-based watermarking algorithms in this project require pretrained model weights.  
Please follow the instructions below to download the corresponding weights and place them in:

#### • DwTDCT, RivaGAN
DwTDCT is a classical, non–neural-network watermarking algorithm, while RivaGAN is a neural watermarking method.  
Both algorithms are integrated in this project by directly calling the Python API provided by: https://github.com/ShieldMnt/invisible-watermark.git

#### • HiDDeN  
Download from: https://github.com/ando-khachatryan/HiDDeN.git

#### • StegaStamp  
Download from: https://github.com/ningyu1991/ArtificialGANFingerprints.git

#### • Stable Signature  
Download from: https://github.com/facebookresearch/stable_signature

#### • Vine
This watermarking algorithm does not require manual weight preparation.  
When invoked for the first time, it will automatically download the required model weights from HuggingFace.

📁 Example Directory Structure (after downloading all weights)

```
algorithms/
└── checkpoints/
	├── hidden/
	│   ├── combined-noise--epoch-400.pyt
	│   ├── crop-epoch-300.pyt
	│   └── no-noise--epoch-400.pyt
	├── stable_signature/
	│   ├── dec_48b_whit.torchscript.pt
	│   ├── sd2_decoder.pth
	│   └── v2-1_512-ema-pruned.ckpt
	└── stegastamp/
		├── AFHQ_cat2dog_256x256_decoder.pth
		└── AFHQ_cat2dog_256x256_encoder.pth
```


---

### 3. Usage

WMSuite provides two basic functionalities: watermark embedding and watermark extraction.  
You can run them directly using the provided shell scripts:

To embed watermarks into all images within a specified folder, run:
```bash
bash emb.sh
```
To extract watermarks from images in a specified folder, run:
```bash
bash extract.sh
```


### 🙏 Acknowledgements

This project integrates implementations and pretrained models from several existing watermarking algorithms.  
We would like to acknowledge and thank the authors of the following works and repositories:

---

#### **• HiDDeN**  
Repository: https://github.com/ando-khachatryan/HiDDeN.git  

#### **• StegaStamp**  
Repository: https://github.com/ningyu1991/ArtificialGANFingerprints.git  

#### **• Stable Signature (Meta AI)**  
Repository: https://github.com/facebookresearch/stable_signature  

#### **• Vine**  
Repository: https://github.com/Shilin-LU/VINE.git  

#### **• DwTDCT & RivaGAN**  
Integrated via the Python API provided by the Invisible Watermark library:  
https://github.com/ShieldMnt/invisible-watermark.git  

---

All copyrights, model weights, and algorithmic contributions belong to their respective authors.  
WMSuite only provides a unified interface to access these methods for research and experimentation.





