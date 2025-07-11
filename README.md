# DCGAN with Automatic Mixed Precision (AMP) — PyTorch Implementation

This repository provides a clean and optimized implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch, designed to train efficiently with **Automatic Mixed Precision (AMP)** on mid-tier GPUs (e.g., NVIDIA RTX 3050 6GB). The model is trained on the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for realistic face generation.

## 🔧 Features

- ✅ PyTorch-based DCGAN architecture
- ⚡ Optimized with `torch.cuda.amp` for mixed precision training
- 🧠 Stable training using `BCEWithLogitsLoss` (eliminates Sigmoid instability)
- 📦 Supports raw `.jpg` image folders (no class subdirectories required)
- 💾 Saves sample outputs after every epoch
- 🖥️ Fully compatible with Windows + CUDA (no multiprocessing issues)

## 📁 Project Structure

```
📦 DCGAN-AMP-CelebA/
├── celebA.py              # Main training script
├── output/                # Generated image samples per epoch
├── README.md              # Project documentation
└── requirements.txt       # Optional: Dependencies list
```

## 🧱 Model Overview

### Generator
- Input: 100-dimensional latent vector
- Architecture: Transposed Convolutions + BatchNorm + ReLU
- Output: 64×64×3 image (RGB), normalized via Tanh

### Discriminator
- Input: 64×64×3 image
- Architecture: Convolutions + BatchNorm + LeakyReLU
- Output: Real/fake logits (no Sigmoid)

## 📥 Dataset

This project uses the **CelebA** dataset.

### Download Instructions:
1. [Create a Kaggle account](https://www.kaggle.com)
2. Download from: [https://www.kaggle.com/datasets/jessicali9530/celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
3. Extract the archive and place the image folder here:

```
D:/Name&Entity_project/data/img_align_celeba/
```

> The folder should contain raw `.jpg` images (e.g., 000001.jpg, 000002.jpg, ...)

## 🚀 Training

To start training, run:

```bash
python celebA.py
```

Sample images will be saved to the `output/` directory after each epoch.

## 💡 Technical Notes

- Uses `torch.cuda.amp.GradScaler()` for mixed precision (safe for PyTorch < 2.3.0)
- All training logic wrapped in `if __name__ == "__main__"` (Windows-safe multiprocessing)
- Discriminator outputs **raw logits** for compatibility with `BCEWithLogitsLoss`
- Generator uses fixed noise vectors for consistent sample generation

## 🖼️ Sample Outputs

After each epoch, a grid of generated face images will be saved like:

```
output/
├── fake_epoch_1.png
├── fake_epoch_5.png
├── fake_epoch_25.png
```

## 🔗 References

- [DCGAN Paper — Radford et al. (2015)](https://arxiv.org/abs/1511.06434)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Automatic Mixed Precision — PyTorch Docs](https://pytorch.org/docs/stable/amp.html)

## 👤 Author

**Yasir Farooqui**  
MSc AI & ML | Researcher | Developer  
📫 [LinkedIn](https://www.linkedin.com/) | ✉️ [your-email@example.com]

## 📜 License

This project is licensed under the MIT License. Feel free to use and modify for academic or commercial purposes.