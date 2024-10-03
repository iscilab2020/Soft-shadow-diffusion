from .third_party.sdf_vae.models import *
from abc import abstractmethod
from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from functools import partial, wraps



def remove_backbone(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vbackbone = hasattr(self, 'backbone')
        if has_vbackbone:
            backbone = self.backbone
            delattr(self, 'backbone')

        out = fn(self, *args, **kwargs)

        if has_vbackbone:
            self.backbone = backbone

        return out
    return inner


class VAEPointCloudSDFModel(nn.Module):
    """
    Encode point clouds using a transformer, and query points using cross
    attention to the encoded latents.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        vae=0,
        with_ae=True,
    ):
        super().__init__()
        self._device = device

        self.specs = {"SdfModelSpecs" : {
                "hidden_dim" : 128,
                "latent_dim" : 128,
                "pn_hidden_dim" : 128,
                "num_layers" : 12,
                "vae":with_ae
                },
                    
                "SampPerMesh" : 48000,
                "PCsize" : 1024,
            
                "kld_weight" : 3e-1 if int(vae) else 3e-1,
                "latent_std" : 1. if int(vae) else "zero_mean" ,
            
                "sdf_lr" : 1e-4,
                "training_task":"modulation",
                }
        
        self.model =  CombinedModel(self.specs)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def default_batch_size(self) -> int:
        return self.n_query

    def encode_point_clouds(self, point_clouds: torch.Tensor, return_distribution=False):
        plane_features = self.model.sdf_model.pointnet.get_plane_features(point_clouds)
        mu, logvar = self.model.vae_model.encode(torch.cat(plane_features, dim=1))
        if return_distribution:
            return [mu, logvar]
        return self.model.vae_model.reparameterize(mu, logvar)
    
    def encode_to_triplane(self, point_clouds: torch.Tensor):
        plane_features = self.model.sdf_model.pointnet.get_plane_features(point_clouds)
        return torch.cat(plane_features, 1)
        
    
    def get_loss(self, query, point_clouds, sdf, epoch, reduction="none"):
        x = {"xyz":query, "gt_sdf":sdf, "point_cloud":point_clouds, "epoch":epoch, "reduction":reduction}
        return self.model.train_modulation(x)
        
    
    def predict_occupancy(
        self, x: torch.Tensor, encoded:torch.Tensor
    ) -> torch.Tensor:
        encoded =  self.model.vae_model.decode(encoded) #if len(encoded.shape) <=2 else self.model.vae_model(encoded)[0]
        pred_sdf = self.model.sdf_model.forward_with_plane_features(encoded, x)
        return pred_sdf
    
    
    def forward(
        self,
        x: torch.Tensor,
        point_clouds: torch.Tensor,
        sdf:torch.Tensor,
        epoch=100
    ) -> torch.Tensor:
        return self.get_loss(x, point_clouds, sdf, epoch=epoch)
    
    
    @remove_backbone
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_backbone
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)
    
    