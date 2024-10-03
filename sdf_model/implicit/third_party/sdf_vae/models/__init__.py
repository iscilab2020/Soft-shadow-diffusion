#!/usr/bin/python3

from implicit.third_party.sdf_vae.models.sdf_model import SdfModel
from implicit.third_party.sdf_vae.models.autoencoder import BetaVAE, StraightTrough, AE

from implicit.third_party.sdf_vae.models.archs.encoders.conv_pointnet import UNet

# from third_party.sdf_vae.models.diffusion import *
# from third_party.sdf_vae.models.archs.diffusion_arch import * 
#from diffusion import *
from implicit.third_party.sdf_vae.models.sdf_model import SdfModel

from implicit.third_party.sdf_vae.models.combined_model import CombinedModel