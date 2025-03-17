import torch
import torch.nn as nn
import torch.nn.functional as F  
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional import structural_similarity_index_measure
from facenet_pytorch import InceptionResnetV1
from modules.denormalize import denormalize_bin, denormalize_tr, denormalize_ar
    
class Evaluation(nn.Module):
    def __init__(self):
        super(Evaluation, self).__init__()
        self.fid_weight = 100.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fid_metric = FrechetInceptionDistance(feature=64).to(self.device)
        self.fid_metric.reset()
        self.facenet = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        for param in self.facenet.parameters():
            param.requires_grad = False  # Freeze the model
        self.cosloss = nn.CosineEmbeddingLoss()

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
        idsim = self.idsimilarity(real, fake)
        ssim = structural_similarity_index_measure(fake, real)
        real_tr = denormalize_tr(r)
        fake_tr = denormalize_tr(f)
        self.fid_metric.update(real_tr, real=True)
        self.fid_metric.update(fake_tr, real=False)
        fidloss = self.fid_metric.compute()
        fid = self.fid_weight * fidloss

        components = {
            "FID": fid.item(),
            "ID-SIM": idsim.item(),
            "SSIM": ssim.item()
        }

        return components
