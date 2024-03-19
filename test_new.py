import torch
from torchvision import transforms
from PIL import Image
from model_compat import DSN
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np

# Step 1: Load the trained model
model_path = '/content/DSN/model/dsn_mnist_mnistm_epoch_99.pth'  # Replace X with the epoch number of the trained model
model = DSN()
model.load_state_dict(torch.load(model_path))
model.eval()

# Step 2: Prepare your test data
# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.images = os.listdir(data_root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_root, self.images[idx])
        image = Image.open(img_name).convert('RGB')  # Convert to RGB if needed

        if self.transform:
            image = self.transform(image)

        return image

# Modify the transformation
img_transform = transforms.Compose([
    # transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
# You can use a similar CustomDataset class and DataLoader as you used for training data preparation
# Assuming you have test_loader as your test DataLoader
dataset_test = CustomDataset(data_root='/content/DSN/dataset/leaf_target/test', transform=img_transform)
test_loader = DataLoader(dataset_test, batch_size=12, shuffle=True, num_workers=2)

# Step 3: Pass the test data through the model to obtain reconstructed images
mse_loss = torch.nn.MSELoss()
test_mse = 0.0
num_samples = 0

for images in test_loader:
    # Assuming images are the test images
    with torch.no_grad():
        result = model(input_data=images, mode='target', rec_scheme='all', p=0.0)  # Adjust the arguments as needed
        reconstructed_images = result[-1]  # Assuming reconstructed images are the last element of the result

    # Step 4: Save the original and reconstructed images and calculate MSE
    num_samples += images.size(0)
    for i in range(images.size(0)):
        original_image = transforms.ToPILImage()(images[i].cpu())
        reconstructed_image = transforms.ToPILImage()(reconstructed_images[i].cpu())

        # Save original and reconstructed images
        original_image.save(f'original_{num_samples}.jpg')
        reconstructed_image.save(f'reconstructed_{num_samples}.jpg')

        # Calculate and accumulate MSE
        test_mse += mse_loss(images[i], reconstructed_images[i]).item()

# Step 5: Calculate average MSE
average_mse = test_mse / num_samples
print(f'Average MSE on the test set: {average_mse}')
