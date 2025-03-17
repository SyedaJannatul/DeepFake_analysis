import io
from io import BytesIO
import tempfile
import os
import shutil
from gradio_client import Client, file
import requests
import time
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import pickle
import warnings
import argparse 
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.checkpoint import checkpoint  

from torchvision.models import vgg16
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional import structural_similarity_index_measure
from facenet_pytorch import InceptionResnetV1

from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
from taming.models.vqgan import GumbelVQ

from modules.modelz import DeepfakeToSourceTransformer
from modules.frameworkeval import DF
from modules.segmentface import FaceSegmenter
from modules.denormalize import denormalize_bin, denormalize_tr, denormalize_ar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = Client("felixrosberg/face-swap")

# --------------VQGAN & Model_Z:Paths to config and checkpoints------------------------------
config_path = "models/config.yaml"
checkpoint_path_f = "models/model_vaq_f.pth"
checkpoint_path_g = "models/model_vaq_g.pth"
model_z1_path = "models/model_z1_f.pth"
model_z2_path = "models/model_z2_g.pth"

config = OmegaConf.load(config_path)
# Extract parameters specific to GumbelVQ
vq_params = config.model.params
# Initialize the GumbelVQ models
model_vaq_f = GumbelVQ(
            ddconfig=vq_params.ddconfig,
            lossconfig=vq_params.lossconfig,
            n_embed=vq_params.n_embed,
            embed_dim=vq_params.embed_dim,
            kl_weight=vq_params.kl_weight,
            temperature_scheduler_config=vq_params.temperature_scheduler_config).to(device)
model_vaq_g = GumbelVQ(
            ddconfig=vq_params.ddconfig,
            lossconfig=vq_params.lossconfig,
            n_embed=vq_params.n_embed,
            embed_dim=vq_params.embed_dim,
            kl_weight=vq_params.kl_weight,
            temperature_scheduler_config=vq_params.temperature_scheduler_config).to(device)

##________________________Transformation______________________________

transform = T.Compose([
    T.Resize((256, 256)),   
    T.ToTensor(),         
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  # Normalize to [-1, 1]

#____________________________Main________________________________________
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a deepfake image.")
    parser.add_argument("deepfake_img", type=str, help="Path to the deepfake image file.")
    args = parser.parse_args()

    # Use the provided deepfake image path
    deepfake_img = args.deepfake_img
    # Segment the face
    segmenter = FaceSegmenter(threshold=0.5)
    deepfake_seg = segmenter.segment_face(deepfake_img)
    
    #------------Initialize Models------------------------
    checkpoint_f = torch.load(checkpoint_path_f, map_location=device)
    model_vaq_f.load_state_dict(checkpoint_f, strict=True)
    model_vaq_f.eval()

    checkpoint_g = torch.load(checkpoint_path_g, map_location=device)
    model_vaq_g.load_state_dict(checkpoint_g, strict=True)
    model_vaq_g.eval()

    model_z1 = DeepfakeToSourceTransformer().to(device)
    model_z1.load_state_dict(torch.load(model_z1_path, map_location=device), strict=True)
    model_z1.eval()

    model_z2 = DeepfakeToSourceTransformer().to(device)
    model_z2.load_state_dict(torch.load(model_z2_path, map_location=device), strict=True)
    model_z2.eval()

    criterion = DF()
    
    with torch.no_grad():
        # Load and preprocess input image
        img = Image.open(deepfake_img).convert('RGB')
        segimg = Image.open(deepfake_seg).convert('RGB')
        df_img = transform(img).unsqueeze(0).to(device)  # Shape: (1, 3, 256, 256)
        seg_img = transform(segimg).unsqueeze(0).to(device)
        
        # Calculate quantized_block for all images
        z_df, _, _ = model_vaq_f.encode(df_img) 
        z_seg, _, _ = model_vaq_g.encode(seg_img) 
        rec_z_img1 = model_z1(z_df) 
        rec_z_img2 = model_z2(z_seg) 
        rec_img1 = model_vaq_f.decode(rec_z_img1).squeeze(0)
        rec_img2 = model_vaq_g.decode(rec_z_img2).squeeze(0)
        rec_img1_pil = T.ToPILImage()(denormalize_bin(rec_img1))
        rec_img2_pil = T.ToPILImage()(denormalize_bin(rec_img2))
        rec_img1_pil.save("src1.png")
        rec_img2_pil.save("src2.png")

        # Save PIL images to temporary files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp1, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp2:
            
            rec_img1_pil.save(temp1, format="PNG")
            rec_img2_pil.save(temp2, format="PNG")
            
            temp1_path = temp1.name
            temp2_path = temp2.name

        # Pass file paths to Gradio client
        result = client.predict(
            target=file(temp1_path),
            source=file(temp2_path), slider=100, adv_slider=100,
            settings=["Adversarial Defense"], api_name="/run_inference"
        )

        # Clean up temporary files
        os.remove(temp1_path)
        os.remove(temp2_path)

        # Load result and compute loss
        dfimage_pil = Image.open(result)
        dfimage_pil.save("rec_df.png")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp3:
            dfimage_pil.save(temp3, format="PNG")
            rec_df = denormalize_bin(transform(Image.open(temp3.name))).unsqueeze(0).to(device)
            os.remove(temp3.name)

        rec_loss, _ = criterion(df_img, rec_df)

        print(f"Reconstruction Loss: {rec_loss.item():.3f}")