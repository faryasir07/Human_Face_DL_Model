import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_DIR = r"D:/Name&Entity_project/data/img_align_celeba"
IMAGE_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 25
Z_DIM = 100
LR = 0.0002
BETA1 = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("output", exist_ok=True)

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# ---------------- CUSTOM DATASET ----------------
class RawImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# ---------------- GENERATOR ----------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._block(Z_DIM, 512, 4, 1, 0),
            self._block(512, 256, 4, 2, 1),
            self._block(256, 128, 4, 2, 1),
            self._block(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)

# ---------------- DISCRIMINATOR ----------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._block(3, 64, 4, 2, 1, bn=False),
            self._block(64, 128, 4, 2, 1),
            self._block(128, 256, 4, 2, 1),
            self._block(256, 512, 4, 2, 1),
            nn.Conv2d(512, 1, 4, 1, 0),  # ✅ REMOVED Sigmoid (use logits)
        )

    def _block(self, in_c, out_c, k, s, p, bn=True):
        layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------- TRAINING ----------------
def train():
    dataset = RawImageDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # ✅ AMP-safe loss
    optimizerG = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerD = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

    scaler = torch.cuda.amp.GradScaler()
    fixed_noise = torch.randn(64, Z_DIM, 1, 1, device=DEVICE)

    print("🚀 Training Started...\n")

    for epoch in range(EPOCHS):
        for i, real in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            real = real.to(DEVICE)
            b_size = real.size(0)
            label_real = torch.ones(b_size, device=DEVICE)
            label_fake = torch.zeros(b_size, device=DEVICE)

            # ----- Train Discriminator -----
            D.zero_grad()
            noise = torch.randn(b_size, Z_DIM, 1, 1, device=DEVICE)
            with torch.no_grad():
                fake = G(noise)

            with torch.cuda.amp.autocast():
                out_real = D(real).view(-1)
                out_fake = D(fake).view(-1)
                loss_real = criterion(out_real, label_real)
                loss_fake = criterion(out_fake, label_fake)
                lossD = loss_real + loss_fake

            scaler.scale(lossD).backward()
            scaler.step(optimizerD)
            scaler.update()

            # ----- Train Generator -----
            G.zero_grad()
            noise = torch.randn(b_size, Z_DIM, 1, 1, device=DEVICE)
            with torch.cuda.amp.autocast():
                fake = G(noise)
                out = D(fake).view(-1)
                lossG = criterion(out, label_real)

            scaler.scale(lossG).backward()
            scaler.step(optimizerG)
            scaler.update()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}/{len(dataloader)}] "
                      f"LossD: {lossD.item():.4f} | LossG: {lossG.item():.4f}")

        # Save sample image after every epoch
        with torch.no_grad():
            gen = G(fixed_noise).cpu()
        vutils.save_image(gen, f"output/fake_epoch_{epoch+1}.png", normalize=True)

    print("\n🎉 Training finished. Images saved in 'output/'")

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    train()
