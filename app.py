import io
from io import BytesIO 
import tempfile
import os
import shutil
import requests
import numpy as np
from PIL import Image, ImageOps
import cv2
import math
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import gradio as gr
from modules.modelz import DeepfakeToSourceTransformer
from modules.frameworkeval import DF
from modules.segmentface import FaceSegmenter
from modules.denormalize import denormalize_bin, denormalize_tr, denormalize_ar
from gradio_client import Client, file

client = Client("felixrosberg/face-swap")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = "./models/config.yaml"
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

#_________________Define:Gradio Function________________________

def gen_sources(deepfake_img):
    #----------------DeepFake Face Segmentation-----------------
    segmenter = FaceSegmenter(threshold=0.5)
    img_np = np.array(deepfake_img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    segmented_np = segmenter.segment_face(img_bgr)
    deepfake_seg = Image.fromarray(cv2.cvtColor(segmented_np, cv2.COLOR_BGR2RGB))

    #------------Initialize Models------------------------
    checkpoint_path_f = "./models/model_vaq1_ff.pth"
    checkpoint_f = torch.load(checkpoint_path_f, map_location=device)
    model_vaq_f.load_state_dict(checkpoint_f, strict=True)
    model_vaq_f.eval()

    checkpoint_path_g = "./models/model_vaq2_gg.pth"
    checkpoint_g = torch.load(checkpoint_path_g, map_location=device)
    model_vaq_g.load_state_dict(checkpoint_g, strict=True)
    model_vaq_g.eval()

    model_z1 = DeepfakeToSourceTransformer().to(device)
    model_z1.load_state_dict(torch.load("./models/model_z1_ff.pth", map_location=device), strict=True)
    model_z1.eval()

    model_z2 = DeepfakeToSourceTransformer().to(device)
    model_z2.load_state_dict(torch.load("./models/model_z2_gg.pth", map_location=device), strict=True)
    model_z2.eval()

    criterion = DF()

    with torch.no_grad():
        df_img = transform(deepfake_img.convert('RGB')).unsqueeze(0).to(device)
        seg_img = transform(deepfake_seg).unsqueeze(0).to(device)

        z_df, _, _ = model_vaq_f.encode(df_img)
        z_seg, _, _ = model_vaq_g.encode(seg_img)
        rec_z_img1 = model_z1(z_df)
        rec_z_img2 = model_z2(z_seg)
        rec_img1 = model_vaq_f.decode(rec_z_img1).squeeze(0)
        rec_img2 = model_vaq_g.decode(rec_z_img2).squeeze(0)
        rec_img1_pil = T.ToPILImage()(denormalize_bin(rec_img1))
        rec_img2_pil = T.ToPILImage()(denormalize_bin(rec_img2))


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
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp3:
            dfimage_pil.save(temp3, format="PNG")
            rec_df = denormalize_bin(transform(Image.open(temp3.name))).unsqueeze(0).to(device)
            os.remove(temp3.name)

        rec_loss, _ = criterion(df_img, rec_df)

        return (rec_img1_pil, rec_img2_pil, dfimage_pil, round(rec_loss.item(), 3))

#________________________Create the Gradio interface_________________________________
interface = gr.Interface(
    fn=gen_sources,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=[
        gr.Image(type="pil", label="Recovered Source Image 1 (Target Image)"),
        gr.Image(type="pil", label="Recovered Source Image 2 (Source Image)"),
        gr.Image(type="pil", label="Reconstructed Deepfake Image"),
        gr.Number(label="Reconstruction Loss")
    ],
    examples = ["./images/df1.jpg","./images/df2.jpg","./images/df3.jpg","./images/df4.jpg"],
    theme = gr.themes.Soft(),
    title="Uncovering Deepfake Image for Identifying Source Images",
    description="Upload an DeepFake image.",
)

interface.launch(debug=True)