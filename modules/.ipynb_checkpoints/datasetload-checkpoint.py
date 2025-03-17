import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Define custom dataset for train, validation
class DeepfakeDataset(Dataset):
    def __init__(self, deepfake_dir, source_dir1, transform):
        self.deepfake_dir = deepfake_dir
        self.source_dir1 = source_dir1
        self.transform = transform

        # List all deepfake image filenames
        self.deepfake_images = os.listdir(deepfake_dir)

    def __len__(self):
        return len(self.deepfake_images)

    def __getitem__(self, idx):
        # Get the deepfake image filename
        deepfake_img_name = self.deepfake_images[idx]
        deepfake_img_path = os.path.join(self.deepfake_dir, deepfake_img_name)

        # Extract the source indices from the deepfake filename
        src_idx1, _ = deepfake_img_name.split('.')[0].split('_')
        #_, src_idx2 = deepfake_img_name.split('.')[0].split('_')

        # Load the corresponding source images
        source_img1_path = os.path.join(self.source_dir1, f"{src_idx1}.png")
        deepfake_img = Image.open(deepfake_img_path).convert('RGB')
        source_img1 = Image.open(source_img1_path).convert('RGB')
        
        if self.transform:
            deepfake_img = self.transform(deepfake_img)
            source_img1 = self.transform(source_img1)

        # Return the deepfake image and the tuple of source image
        return deepfake_img,source_img1

## Define custom dataset for test
class DeepfakeDatasettest(Dataset):
    def __init__(self, deepfake_dir, source_dir1, transform):
        self.deepfake_dir = deepfake_dir
        self.source_dir1 = source_dir1
        #self.source_dir2 = source_dir2
        self.transform = transform

        # List all deepfake image filenames
        self.deepfake_images = os.listdir(deepfake_dir)

    def __len__(self):
        return len(self.deepfake_images)

    def __getitem__(self, idx):
        # Get the deepfake image filename
        deepfake_img_name = self.deepfake_images[idx]
        deepfake_img_path = os.path.join(self.deepfake_dir, deepfake_img_name)

        # Extract the source indices from the deepfake filename
        img_name = os.path.splitext(deepfake_img_name)[0]  # Remove the file extension
        src_idx1, _ = map(int, deepfake_img_name.split('.')[0].split('_'))

        # Load the corresponding source images
        source_img1_path = os.path.join(self.source_dir1, f"{src_idx1}.png")
        #source_img2_path = os.path.join(self.source_dir2, f"{src_idx2}.png")
        deepfake_img = Image.open(deepfake_img_path).convert('RGB')
        source_img1 = Image.open(source_img1_path).convert('RGB')
        #source_img2 = Image.open(source_img2_path).convert('RGB')
        
        if self.transform:
            deepfake_img = self.transform(deepfake_img)
            source_img1 = self.transform(source_img1)
            #source_img2 = self.transform(source_img2)

        # Return the deepfake image and the tuple of source images
        return deepfake_img, img_name, source_img1
