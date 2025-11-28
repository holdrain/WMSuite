# 💧 WMSuite: Image Watermark Processing Toolkit

> An Python toolset designed for the embedding and extraction of several invisible image watermarks.

## 🚀 Quick Start Guide

### 1. Environment Setup




### 2. Model Weights Configuration 📥

Deep-learning based algorithms require pre-trained model weights. **You will need to manually obtain these files from the original project repositories.** Based on the `METHOD` you choose, place the corresponding files in the recommended directory.

| Algorithm (`METHOD`) | Model Description | **Acquisition Method / Download Guidance** | Recommended Path |
| :--- | :--- | :--- | :--- |
| `vine` | Pre-trained weights for the Vine model. | Please visit the **[Invisible Watermark original repository](https://github.com/ShieldMnt/invisible-watermark)** Releases or documentation to find the weight file. | `./wm/models/vine_weights.pth` |
| `hidden` | Pre-trained weights for the HiDDeN model. | Please visit the **[HiDDeN original repository](https://github.com/ando-khachatryan/HiDDeN)** documentation or data download links to obtain the file. | `./wm/models/hidden_weights.pth` |

---

## ⚙️ Usage and Configuration

### Step A: Watermark Embedding (`emb.sh`)

Used to embed watermark information into images within the source dataset.

# Run the embedding script
bash emb.sh


| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `METHOD` | `"vine"` | The watermarking algorithm to be used. |
| `DATASETS` | `.../train` | **Absolute or relative path to the source dataset.** |
| `OUTPUT_DIR` | `./outputs` | Directory to save the watermarked images. |
| `NUM` | `8` | The total number of images scheduled for processing. |
| `BATCH_SIZE` | `8` | Size of the processing batch. |

### Step B: Watermark Extraction and Detection (`extract.sh`)

Used to extract the watermark from watermarked images and generate a detection report.

# Run the extraction script
bash extract.sh


| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `METHOD` | `"hidden"` | The extraction algorithm (must match the embedding algorithm). |
| `DATAPATH` | `.../outputs/hidden` | **Input path for images to be tested** (usually the output of Step A). |
| `LOG_DIR` | `./outputs` | Directory to save the extraction results (including the detection rate `.txt` file). |
| `DEVICE` | `"cuda:0"` | Specifies the running device (supports `cuda:X` or `cpu`). |
| `BETA` | `1e-6` | Algorithm-specific hyperparameter. |

---

## 📂 Project Structure

WMSuite/
├── wm/                 # Core source package
│   ├── cli/            # Command Line Interface (emb, extract)
│   ├── algorithms/     # Algorithm implementations
│   └── models/         # Models and weights
├── outputs/            # Default output directory
├── emb.sh              # Batch watermark embedding script
├── extract.sh          # Batch watermark extraction script
└── README.md           # Documentation file

---

## 💖 Acknowledgements and Legal Disclaimer

This project utilizes code and ideas referenced from the following excellent open-source projects. In accordance with the MIT and similar permissive licenses, you **must** retain the original project's copyright notice and license text when reusing the code.

### Referenced Projects

| Project Name | GitHub Link | License | Legal Requirement |
| :--- | :--- | :--- | :--- |
| **Invisible Watermark** | [ShieldMnt/invisible-watermark](https://github.com/ShieldMnt/invisible-watermark) | MIT | Must retain original copyright and license. |
| **HiDDeN** | [ando-khachatryan/HiDDeN](https://github.com/ando-khachatryan/HiDDeN) | MIT | Must retain original copyright and license. |
| **ArtificialGANFingerprints** | [ningyu1991/ArtificialGANFingerprints](https://github.com/ningyu1991/ArtificialGANFingerprints) | MIT | Must retain original copyright and license. |
| **Stable Signature** | [facebookresearch/stable_signature](https://github.com/facebookresearch/stable_signature) | MIT | Must retain original copyright and license. |

**We highly recommend adding a brief author and source declaration at the top of any source code files you have directly copied or modified.**
```