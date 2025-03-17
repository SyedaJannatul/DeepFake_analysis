import os
import shutil
import requests
import time
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import pickle
import warnings
import argparse 
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.checkpoint import checkpoint  

from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
from taming.models.vqgan import GumbelVQ

from datasetload import DeepfakeDatasettest
from models.finetunedvqgan import generator
from models.modelz import DeepfakeToSourceTransformer
from trainvalidation import test_z

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------VQGAN & Model_Z:Paths to config and checkpoints------------------------------
config_path = "models/ckpt/config.yaml"
checkpoint_path_f = "models/ckpt/model_vaq_f.pth"
checkpoint_path_g = "models/ckpt/model_vaq_g.pth"
model_z1_path = "models/ckpt/model_z1_f.pth"
model_z2_path = "models/ckpt/model_z2_g.pth"
 
    
#-----------------------------Main-------------------------------------------------------------
if __name__ == "__main__":
    # Define image transformations
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor(),         
                        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  # Normalize to [-1, 1]
    batch_size = 8
    learning_rate = 5e-5
    num_epochs = 20

    # Create the dataset and dataloader for test
    deepfake_dir_test = '../dataset/test/model_z1/ffhq_celeba_e4s/df'
    source_dir1_test = '../dataset/test/model_z1/ffhq_celeba_e4s/src'
    #source_dir2_test = '../dataset/test/src_remb'
    dataset_test = DeepfakeDatasettest(deepfake_dir_test,source_dir1_test,transform=transform)
    dataloader_test = DataLoader(dataset_test, batch_size=batch,shuffle=False)
    
    # Initialize the VQGAN class & Load the decoder models
    vqgan = generator(config_path, checkpoint_path_f, checkpoint_path_g, device=device)
    model_vaq_f, _ = vqgan.load_models()
    #_, model_vaq_g = vqgan.load_models()
    print("Decoder loaded successfully!")
    
    # Initialize the model_z class & Load the quantized block models
    model_z1 = DeepfakeToSourceTransformer().to(device)
    model_z1.load_state_dict(torch.load(model_z1_path, map_location=device), strict=True)
    #model_z2 = DeepfakeToSourceTransformer().to(device)
    #model_z2.load_state_dict(torch.load(model_z2_path, map_location=device), strict=True)
    print("Latent Model loaded successfully!")

    # test
    msg = test_z(model_vaq_f, model_z1, dataloader_test)    

    