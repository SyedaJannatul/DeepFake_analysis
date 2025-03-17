import torch
import torch.nn as nn
import torch.nn.functional as F  
from torchvision.models import vgg16
from torchmetrics.functional import structural_similarity_index_measure
from facenet_pytorch import InceptionResnetV1
from modules.denormalize import denormalize_bin, denormalize_tr, denormalize_ar
    
class CompositeLoss(nn.Module):
    def __init__(self,mse_w=0.25):
        super(CompositeLoss, self).__init__()
        self.mse_weight = mse_w
        self.perceptual_weight = 0.25
        self.ssim_weight = 0.25
        self.idsim_weight = 0.25
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = vgg16(pretrained=True).features[:16].to(self.device).eval()
        self.facenet = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        for param in self.facenet.parameters():
            param.requires_grad = False  # Freeze the model
        self.cosloss = nn.CosineEmbeddingLoss()
   
    def perceptual_loss(self, real, fake):
        with torch.no_grad():  # VGG is frozen during training
            real_features = self.vgg(real)
            fake_features = self.vgg(fake)
        return F.mse_loss(real_features, fake_features)

    def idsimilarity(self, real, fake):
        with torch.no_grad():
            # Extract embeddings
            input_embed = self.facenet(real).to(self.device)
            generated_embed = self.facenet(fake).to(self.device)
        # Compute cosine similarity loss
        target = torch.ones(input_embed.size(0)).to(self.device)  # Target = 1 (maximize similarity)
        return self.cosloss(input_embed, generated_embed, target)

    def forward(self, r, f):
        real = denormalize_bin(r) #[-1,1] to [0,1]
        fake = denormalize_bin(f)
        mse_loss = F.mse_loss(real, fake)
        perceptual_loss = self.perceptual_loss(real, fake)
        idsim_loss = self.idsimilarity(real, fake)
        ssim = structural_similarity_index_measure(fake, real)
        ssim_loss = 1 - ssim
        id_si = 1 - idsim_loss

        total_loss = (self.mse_weight * mse_loss) + (self.perceptual_weight * perceptual_loss) + (self.idsim_weight * idsim_loss) + (self.ssim_weight * ssim_loss)
        components = {
            "MSE Loss": mse_loss.item(),
            "Perceptual Loss": perceptual_loss.item(),
            "ID-SIM Loss": idsim_loss.item(),
            "SSIM Loss": ssim_loss.item()
        }

        return total_loss, components
