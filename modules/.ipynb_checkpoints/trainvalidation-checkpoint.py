import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.checkpoint import checkpoint  

from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
from taming.models.vqgan import GumbelVQ

from modules.modelz import DeepfakeToSourceTransformer
from modules.compositeloss import CompositeLoss
from modules.evaluation import Evaluation
from modules.denormalize import denormalize_bin, denormalize_tr, denormalize_ar

def hinge_loss_discriminator(disc_real, disc_fake):
    real_loss = torch.mean(torch.relu(1 - disc_real))  # Push real outputs ≥ 1
    fake_loss = torch.mean(torch.relu(1 + disc_fake))  # Push fake outputs ≤ -1
    return (real_loss + fake_loss) / 2

#--------------------------------------------------------------------------------------------
#----------------------------------------validation--------------------------------------------
def valid_z(model_vaq, model_z1, discriminator, dataloader):
    loss_fn = CompositeLoss(mse_w=0.25)
    loss_z = CompositeLoss(mse_w=0.0)
    score = Evaluation()
    model_vaq.eval()
    model_z1.eval()
    discriminator.eval()
    
    with torch.no_grad():
        epoch_loss = 0.0
        epoch_zloss = 0.0
        epoch_disc_loss = 0.0
        epoch_components = {
            "MSE Loss": 0.0,
            "Perceptual Loss": 0.0,
            "ID-SIM Loss": 0.0,
            "SSIM Loss": 0.0
            }
        epoch_scores = {
            "FID": 0.0,
            "ID-SIM": 0.0,
            "SSIM": 0.0
        }
        
        for df_img,img1 in dataloader:
            df_img = df_img.to(device)
            img1 = img1.to(device)
            
            #calculate quantized_block for all images
            z_df,_,_ = model_vaq.encode(df_img) 
            z_img1,_,_ = model_vaq.encode(img1) 
            rec_z_img1 = model_z1(z_df)
            rec_img1 = model_vaq.decode(rec_z_img1).to(device)
            # Quantized block Reconstruction loss
            zl = F.mse_loss(z_img1, rec_z_img1)
            loss, _ = loss_z(img1, rec_img1)
            z_loss = (z1*0.25) + loss
            # Image Reconstruction loss
            total_loss, components = loss_fn(img1, rec_img1) # reconstruction loss  
            all_scores = score(img1, rec_img1)
            # Discriminator loss
            disc_real = discriminator(img1)  # Real images
            disc_fake = discriminator(rec_img1.detach())  # Stop gradient from affecting generator
            d_loss = hinge_loss_discriminator(disc_real, disc_fake)

            # Update losses image
            epoch_zloss += z_loss.item()
            epoch_loss += total_loss.item()
            epoch_disc_loss += d_loss.item()
            for key in epoch_components:
                epoch_components[key] += components[key]
            for key in epoch_scores:
                epoch_scores[key] += all_scores[key]


        ##---------------------Validation summary------------------------------------
        
        # Average epoch losses image
        epoch_zloss /= len(dataloader)
        epoch_loss /= len(dataloader)
        epoch_disc_loss /= len(dataloader)
        for key in epoch_components:
            epoch_components[key] /= len(dataloader)
        for key in epoch_scores:
            epoch_scores[key] /= len(dataloader)
        
        print("Validation:Losses")
        print(f"Quantized:{epoch_zloss:.4f}; MSE:{epoch_components['MSE Loss']:.4f}; Perceptual:{epoch_components['Perceptual Loss']:.4f}; ID-SIM:{epoch_components['ID-SIM Loss']:.4f}; SSIM:{epoch_components['SSIM Loss']:.4f}; Discrim:{epoch_disc_loss:.4f};")
        print("Validation:Scores")
        print(f"FID:{epoch_scores['FID']:.4f}; ID-SIM Score:{epoch_scores['ID-SIM']:.4f}; SSIM score:{epoch_scores['SSIM']:.4f}; ")
    return "Done!!"

#--------------------------------------------------------------------------------------------
#----------------------------------------training--------------------------------------------
def train_z(model_vaq, model_z1, discriminator, dataloader, dataloader_valid, lrs, num_epochs):
    loss_fn = CompositeLoss(mse_w=0.25)
    loss_z = CompositeLoss(mse_w=0.0)
    score = Evaluation()
    model_z1.train()
    model_vaq.train()
    discriminator.train()
    
    for param in model_vaq.encoder.parameters():
        param.requires_grad = False
        
    optimizer1 = optim.AdamW(model_z1.parameters(), lr=lrs)
    optimizer2 = optim.AdamW(model_vaq.parameters(), lr=4.5e-6)
    optimizer_disc = optim.AdamW(discriminator.parameters(), lr=lrs)

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        epoch_zloss = 0.0
        epoch_disc_loss = 0.0        
        epoch_components = {
            "MSE Loss": 0.0,
            "Perceptual Loss": 0.0,
            "ID-SIM Loss": 0.0,
            "SSIM Loss": 0.0
            }
        epoch_scores = {
            "FID": 0.0,
            "ID-SIM": 0.0,
            "SSIM": 0.0
        }

        for df_img,img1 in dataloader:
            z_df,_,_ = model_vaq.encode(df_img) 
            z_img1,_,_ = model_vaq.encode(img1) 
            rec_z_img1 = model_z1(z_df)
            rec_img1 = model_vaq.decode(rec_z_img1).to(device)
            
            # Quantized block Reconstruction loss
            zl = F.mse_loss(z_img1, rec_z_img1)
            loss, _ = loss_z(img1, rec_img1)
            z_loss = (z1*0.25) + loss
            optimizer1.zero_grad()
            z_loss.backward()
            optimizer1.step()
            
            # VQGAN-decoder loss, Backward pass and optimizer step
            total_loss, components = loss_fn(img1, rec_img1)
            all_scores = score(img1, rec_img1) 
            disc_fake = discriminator(rec_img1) # Adversarial loss
            g_adv_loss = -torch.mean(disc_fake)  # Generator wants to maximize D(fake)
            decoder_loss = total_loss + (0.01 * g_adv_loss)
            optimizer2.zero_grad()
            decoder_loss.backward()
            optimizer2.step()
            
            ## --------------------- Discriminator Training ---------------------
            disc_real = discriminator(img1)  # Real images
            disc_fake = discriminator(rec_img1.detach())  # Stop gradient from affecting generator

            # Hinge loss for discriminator
            d_loss = hinge_loss_discriminator(disc_real, disc_fake)

            # Backward pass and optimizer step
            optimizer_disc.zero_grad()
            d_loss.backward()
            optimizer_disc.step()
                        
            # Update losses image
            epoch_zloss += z_loss.item()
            epoch_loss += total_loss.item()
            epoch_disc_loss += d_loss.item()
            for key in epoch_components:
                epoch_components[key] += components[key]
            for key in epoch_scores:
                epoch_scores[key] += all_scores[key]


        ##---------------------Training summary------------------------------------
        
        # Average epoch losses image
        epoch_zloss /= len(dataloader)
        epoch_loss /= len(dataloader)
        epoch_disc_loss /= len(dataloader)
        for key in epoch_components:
            epoch_components[key] /= len(dataloader)
        for key in epoch_scores:
            epoch_scores[key] /= len(dataloader)


        if (epoch + 1) % 10 == 0:
            path_z1 = "models/z1/model_z1_" + str(epoch) + ".pth"
            path_vaq = "models/vaq1/model_vaq1_" + str(epoch) + ".pth"
            #path_dis = "models/dis1/model_dis1_" + str(epoch) + ".pth"
            torch.save(model_z1.state_dict(), path_z1)
            torch.save(model_vaq.state_dict(), path_vaq)
            #torch.save(discriminator.state_dict(), path_dis)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print("Training:Losses")
        print(f"Quantized:{epoch_zloss:.4f}; MSE:{epoch_components['MSE Loss']:.4f}; Perceptual:{epoch_components['Perceptual Loss']:.4f}; ID-SIM:{epoch_components['ID-SIM Loss']:.4f}; SSIM:{epoch_components['SSIM Loss']:.4f}; Discrim:{epoch_disc_loss:.4f};")
        print("Traning:Scores")
        print(f"FID:{epoch_scores['FID']:.4f}; ID-SIM Score:{epoch_scores['ID-SIM']:.4f}; SSIM score:{epoch_scores['SSIM']:.4f}; ")
    
        msg = valid_z(model_vaq, model_z1, discriminator, dataloader_valid)
        epoch_time = (time.time() - start_time) / 3600
        print(f"Time:{epoch_time:.2f} hrs")

    return msg   

#--------------------------------------------------------------------------------------------
#------------------------------------test------------------------------------------------
def test_z(model_vaq, model_z1, dataloader):
    output_folder = 'modelf_gen/ffhq_celeba_e4s/'
    os.makedirs(output_folder, exist_ok=True)
    model_vaq.eval()
    model_z1.eval()
    loss_fn = CompositeLoss(mse_w=0.25)
    score = Evaluation()
    
    with torch.no_grad():
        epoch_loss = 0.0
        epoch_components = {
            "MSE Loss": 0.0,
            "Perceptual Loss": 0.0,
            "ID-SIM Loss": 0.0,
            "SSIM Loss": 0.0
            }
        epoch_scores = {
            "FID": 0.0,
            "ID-SIM": 0.0,
            "SSIM": 0.0
        }
        for df_img,names,img1 in dataloader:
            df_img = df_img.to(device)
            img1 = img1.to(device)
            
            #calculate quantized_block for all images
            z_df,_,_ = model_vaq.encode(df_img) 
            rec_z_img1 = model_z1(z_df) 
            rec_img1 = model_vaq.decode(rec_z_img1)
            total_loss, components = loss_fn(img1, rec_img1)
            all_scores = score(img1, rec_img1)

            # Update losses image
            epoch_loss += total_loss.item()
            for key in epoch_components:
                epoch_components[key] += components[key]
            for key in epoch_scores:
                epoch_scores[key] += all_scores[key]

            # Denormalize decoded images
            de_img_arr = denormalize_ar(rec_img1)

            # Save each image in the batch
            for i in range(de_img_arr.shape[0]):
                img_pil = Image.fromarray(de_img_arr[i].astype('uint8'))
                image_name = f"src1_{names[i]}.png" 
                #image_name = f"src2_{names[i]}.png" 
                image_path = os.path.join(output_folder, image_name)
                img_pil.save(image_path)
                print(f"Saved: {image_path}")


        ##---------------------Test summary------------------------------------
        
        # Average epoch losses image
        epoch_loss /= len(dataloader)
        for key in epoch_components:
            epoch_components[key] /= len(dataloader)
        for key in epoch_scores:
            epoch_scores[key] /= len(dataloader)
        
        print("Test:Losses")
        print(f"Total:{epoch_loss:.4f}; MSE:{epoch_components['MSE Loss']:.4f}; Perceptual:{epoch_components['Perceptual Loss']:.4f}; ID-SIM:{epoch_components['ID-SIM Loss']:.4f}; SSIM:{epoch_components['SSIM Loss']:.4f};")
        print("Test:Scores")
        print(f"FID:{epoch_scores['FID']:.4f}; ID-SIM Score:{epoch_scores['ID-SIM']:.4f}; SSIM score:{epoch_scores['SSIM']:.4f}; ")
    
    return "Done!!"