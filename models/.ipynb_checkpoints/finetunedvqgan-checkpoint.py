import torch
from torch.utils.checkpoint import checkpoint  
from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
from taming.models.vqgan import GumbelVQ

class Generator:
    def __init__(self, config_path, device=device):
        self.config_path = config_path
        self.device = device

    def load_models(self):
        # Load configuration
        config = OmegaConf.load(self.config_path)
        # Extract parameters specific to GumbelVQ
        vq_params = config.model.params
        # Initialize the GumbelVQ models
        model_vaq = GumbelVQ(
            ddconfig=vq_params.ddconfig,
            lossconfig=vq_params.lossconfig,
            n_embed=vq_params.n_embed,
            embed_dim=vq_params.embed_dim,
            kl_weight=vq_params.kl_weight,
            temperature_scheduler_config=vq_params.temperature_scheduler_config,
        ).to(self.device)

        return model_vaq


