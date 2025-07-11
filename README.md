📦 DCGAN with AMP on CelebA Dataset (PyTorch)
A Deep Convolutional GAN (DCGAN) built using PyTorch, trained on the CelebA dataset.
This version uses Automatic Mixed Precision (AMP) to speed up training and reduce GPU memory usage.
Perfect for running on mid-range GPUs like RTX 3050 (6GB).

📌 Features
✅ PyTorch implementation of DCGAN

⚡ Uses AMP (torch.cuda.amp) for faster training

🧠 Stable BCEWithLogitsLoss (no Sigmoid in discriminator)

🎨 Saves generated samples after each epoch

🖼️ Works with raw .jpg images (no class folders)

🖥️ Compatible with Windows

🧱 Architecture
Generator:

Input: random noise (100-dim)

4 transpose conv layers + batch norm + ReLU

Output: 64×64×3 RGB image

Discriminator:

Input: 64×64×3 image

4 conv layers + batch norm + LeakyReLU

Output: single scalar (real/fake logits)

🧰 Requirements
Install dependencies with:

bash
Copy code
pip install torch torchvision tqdm pillow
🧠 Recommended: Use a virtual environment (venv, conda, etc.)

📂 Dataset (CelebA)
Download from Kaggle (requires login)

Extract images to:
D:/Name&Entity_project/data/img_align_celeba

The folder should contain raw .jpg images (not subfolders).

🚀 Running the Training
bash
Copy code
python celebA.py
This will:

Train for 25 epochs

Save images to output/ folder like fake_epoch_1.png, fake_epoch_2.png, etc.

📈 Output Samples
Example output after training for a few epochs:

lua
Copy code
output/
├── fake_epoch_1.png
├── fake_epoch_5.png
├── fake_epoch_25.png
Each image contains a 8×8 grid of generated faces.

💡 Notes
⚠️ Training with AMP requires using BCEWithLogitsLoss instead of BCELoss

⚠️ You must not use Sigmoid at the end of the Discriminator

⚙️ Script uses torch.cuda.amp.GradScaler() for PyTorch < 2.3.0 compatibility

📚 References
DCGAN Paper (Radford et al.)

Official PyTorch DCGAN Tutorial

