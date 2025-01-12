# HSRMamba: Efficient Wavelet Stripe State Space Model for Hyperspectral Image Super-Resolution

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

## 📖 Abstract
Single hyperspectral image super-resolution (SR) aims to restore high-resolution images from low-resolution hyperspectral images. While Convolutional Neural Networks (CNNs) are limited in capturing global dependencies, Transformers excel in modeling long-range dependencies but suffer from quadratic computational complexity. In this paper, we propose HSRMamba, a State Space model-based SR network that balances computational efficiency (linear complexity) with effective global feature capture. HSRMamba reduces modal conflicts between high-frequency spatial features and low-frequency spectral features using wavelet decomposition. Additionally, we introduce a strip-based scanning scheme for Visual Mamba to minimize artifacts from global unidirectional scanning. Extensive experiments demonstrate that HSRMamba reduces computational load and model size while outperforming existing methods, achieving state-of-the-art (SOTA) results.

## 📦 Requirements
- Python 3.8+
- PyTorch 1.6+
- CUDA 11.1+

## 📂Dataset
- [Pavia University](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
- [Chikusei](https://naotoyokoya.com/Download.html)
- [Houston](https://hyperspectral.ee.uh.edu/?page_id=459)

## 🛠️Usage
Place the dataset in the dataset directory, and run the following command:
```bash
python main.py 
```

## 🔍 Contact

If you have any questions or suggestions, please submit an Issue or send a email to <lbs23@mails.jlu.edu.cn>.
