import torch.nn as nn
import torch
from occluder.models.occluder_model import *
from occluder.models.crossformer import CrossFormer


class Unet(nn.Module):

    def __init__(self, latent_dim=512, dim=128, learned_variance=False, cond_drop_prob=0.1,
                 device="cpu", image_channel=3, clip_=False) -> None:
        
        super().__init__()
        self.image_channel=image_channel
        self.cond_drop_prob = cond_drop_prob
        
    
        self.model =  Unet1D(
                            
                            dim=dim,
                            init_dim = None,
                            out_dim = None,
                            dim_mults=(1, 2, 4, 8),
                            channels = 3,
                            self_condition = False,
                            resnet_block_groups = 8,
                            learned_variance = learned_variance,
                            learned_sinusoidal_cond = False,
                            random_fourier_features = False,
                            learned_sinusoidal_dim = 16,
                            condition_seq_dim=latent_dim,

                        )
        
        self.channels = 3
        self.self_condition = False
        self.out_dim = self.model.out_dim
        self.self_condition = self.model.self_condition
        
        self.encoder = CrossFormer(
                            dim = (64, 128, 256, 512,),
                            depth = (2, 4, 8, 2),
                            local_window_size = 4,
                            num_classes = latent_dim,
                            channels = image_channel, 
          
        ) 
        # # self.clip = clip_
        # # if clip_:
        # #     self.encoder, _ = clip.load("ViT-B/16")
        # #     self.encoder.requires_grad_(False)
            
        # # else:
        
        # self.encoder =  ModifiedResNet(layers=[1, 2, 4, 8], output_dim=latent_dim, width=64, input_resolution=128, heads=8)
 
    def encode(self, x):
        if self.image_channel==1 and x.size(1)>1:
            x = torch.mean(x, 1, keepdim=True)
        
        x = self.encoder(x)
        return x
    

    def forward(self, x, t, c, l=None):


        if len(c.shape)>3:
            c = self.encode(c)
         
        if self.training: # Classifer Free Gudiance Drop Out In Trainig Mode
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            c = c * mask[:, None].to(c)

        return self.model(x, t, c)

    

class SSD(nn.Module):
    
        def __init__(self, seq_channel=3, latent_dim=512, learned_variance=False, seq_length=2048, 
                     loss_type="l2", schedule="cosine", device="cpu", image_channel=3,
                     timesteps=200) -> None:
            
                super().__init__()

                out = seq_channel*2 if learned_variance else seq_channel

                model = Unet(latent_dim, learned_variance=learned_variance, device=device, image_channel=image_channel)
                self.channels = seq_channel
                self.seq_length = seq_length
                

                if not learned_variance:
                    
                        self.model = GaussianDiffusion1D(model,
                                        seq_length=seq_length,
                                        timesteps = timesteps,
                                        sampling_timesteps = None,
                                        loss_type = loss_type,
                                        objective = 'pred_noise',
                                        beta_schedule = schedule,
                                        ddim_sampling_eta = 0.,
                                        auto_normalize = True,
                                        )
                        
                else:
                        
                        self.model = LearnedGaussianDiffusion(model=model,
                                        seq_length=seq_length,
                                        timesteps = timesteps,
                                        sampling_timesteps = None,
                                        loss_type = loss_type,
                                        objective = 'pred_noise',
                                        beta_schedule = schedule,
                                        ddim_sampling_eta = 0.,
                                        auto_normalize = True,
                                                )
                

        def forward(self, x, y):
            loss = self.model(x, y)
            return loss
        
        def encode(self, img):
            return self.model.model.encode(img)
        

        def sample(self, c, gudiance_scale=0.0, sample_time=200, seq_length=2048, return_all=False):

            if c is None:
                c = torch.zeros(1, 512).to(self.device)

            else:
                c = self.encode(c)
            self = self.eval()
            self.model.sampling_timesteps=sample_time
            self.model.seq_length=seq_length
            
            return self.model.sample(c, gudiance_scale, return_all)
        
        
        
    