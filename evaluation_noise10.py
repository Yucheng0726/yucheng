
import torch
from torchvision import models, transforms
from scipy.linalg import sqrtm
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os

class ImagesDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_paths = []
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):  # Adjust the extensions as needed
                    self.image_paths.append(os.path.join(root, file))
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

def calculate_fid(real_images, generated_images, device):
    model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()
    model.eval()

    def get_activations(images, model, batch_size=32):
        dataloader = DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=2)
        activations = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(device)  # Move batch to GPU
                pred = model(batch)
                activations.append(pred.cpu().numpy())  # Move activations to CPU
        activations = np.concatenate(activations, axis=0)
        return activations

    real_activations = get_activations(real_images, model)
    generated_activations = get_activations(generated_images, model)

    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)

    mu_generated = np.mean(generated_activations, axis=0)
    sigma_generated = np.cov(generated_activations, rowvar=False)

    diff = mu_real - mu_generated
    covmean = sqrtm(sigma_real.dot(sigma_generated))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return fid

def calculate_inception_score(images, device, batch_size=32, splits=10):
    model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    def get_pred(dataloader):
        preds = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(device)  # Move batch to GPU
                pred = model(batch)
                preds.append(F.softmax(pred, dim=1).cpu().numpy())  # Move predictions to CPU
        return np.concatenate(preds, axis=0)

    dataloader = DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=2)
    preds = get_pred(dataloader)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
        kl_div = np.mean(np.sum(kl_div, axis=1))
        scores.append(np.exp(kl_div))
    return np.mean(scores), np.std(scores)

if __name__ == "__main__":
    generated_images_folder = 'output/randomnoise10'  # Folder containing generated images
    
    generated_images = ImagesDataset(generated_images_folder)
    
    # Assume real_images is a dataset of real images you want to compare against
    real_images_folder = 'output/clean'  # Replace with actual folder path
    real_images = ImagesDataset(real_images_folder)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fid = calculate_fid(real_images, generated_images, device)
    print(f"FID: {fid}")

    inception_score, std = calculate_inception_score(generated_images, device)
    print(f"Inception Score: {inception_score} ± {std}")
