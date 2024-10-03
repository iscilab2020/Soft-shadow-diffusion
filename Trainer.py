
from occluder.models.ema_pytorch import EMA
from occluder.utils.utils import *
from occluder.datasets.datasets_occluder import CoreDataset, DataLoader
from occluder.models.ssd import *
from accelerate import Accelerator
import argparse
import wandb
from pathlib import Path
from torch.optim import Adam
import random

# trainer class
class OccluderTrainer(object):
    def __init__(
        self,
        path_to_data,
        *,
        args=None,
        experiment_name="OccluderGen",
        train_batch_size = 16,
        val_batch_size = 64,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 900000,
        ema_update_every = 10,
        train_maximum=-100,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 10,
        num_samples = 4,
        results_folder = './occluder_results',
        amp = False,
        fp16 = False,
        split_batches = True,
        logger = "wandb"
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no',
            log_with=logger
        )

        self.accelerator.native_amp = amp
        model= SSD(3, args.latent_dim, learned_variance=args.learned_variance,
                   seq_length=args.seq_length, device=self.device, image_channel=args.image_channel, timesteps=args.timesteps)
        
        
        
        self.model = model
        self.channels = self.model.channels
        self.loss_meter = AverageMeter()
        self.args=args
    
        # sampling and training hyperparameters
        assert np.sqrt(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.logger = logger
 

        # dataset and dataloader
        self.train_ds = CoreDataset(
               path_to_data=path_to_data, end_data=train_maximum)
        print(f"Number of Training Examples in {experiment_name}: ", len(self.train_ds))
    
       
        dl = DataLoader(self.train_ds, batch_size = train_batch_size, shuffle = True, num_workers = 10)

        if self.accelerator.is_main_process:
            if logger == "wandb":
                self.accelerator.init_trackers(project_name="OccluderGen",)
                self.accelerator.trackers[0].run.name = experiment_name

            self.val_ds = CoreDataset(
                    path_to_data=path_to_data, start_data=train_maximum)
            
            print(f"Number of Validation Examples in {experiment_name}: ", len(self.val_ds))
            
            test_dl = DataLoader(self.val_ds, batch_size = val_batch_size, pin_memory = True, num_workers = 10, drop_last=True)
            self.test_dl = cycle(test_dl)
            
            
      

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        
        
        # optimizer
        self.opt = Adam(self.model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model , self.opt)


    @property
    def device(self):
        return self.accelerator.device
    
  

    def save(self, milestone, val_data=None):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            "args":self.args,
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        if data is not None:
            torch.save(val_data, str(self.results_folder / f'val-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location="cpu")
        
        model =  self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])
        
        opt = self.accelerator.unwrap_model(self.opt)
        opt.load_state_dict(data["opt"])
        
    
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

        self.model, self.opt = self.accelerator.prepare(model , opt)

   
    def log_samples(self, image, true_sample, recons):
        
            pass

   
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.
                
                for _ in range(self.gradient_accumulate_every):
                    
                    data = next(self.dl)
                        
                    condition_seq = awgn(data["measurements"].permute(0, 3, 1, 2).to(device), 20, 100)
                    training_seq = data["pointclouds"].permute(0, 2, 1).to(device)
                    
                    with self.accelerator.autocast():
                        loss = self.model(training_seq, condition_seq)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                   
                    self.accelerator.backward(loss)
                
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()
               
                accelerator.wait_for_everyone()
                

                
                if accelerator.is_main_process:
                 
                    self.ema.update()

                    self.accelerator.log({'train_loss': total_loss},  step=self.step)
                    # print("Exactle At This Point")
                   
                    if self.step % self.save_and_sample_every == 0:
                        
                        self.ema.ema_model.eval()
                      
                        
                        loss = 0.
                        milestone = self.step // self.save_and_sample_every
                        for _ in range((len(self.val_ds)//self.val_batch_size)+1):
                    
                            data = next(self.test_dl)
                         
                            condition_seq = awgn(data["measurements"].permute(0, 3, 1, 2).to(device), 30)
                            training_seq = data["pointclouds"].permute(0, 2, 1).to(device)

                            with torch.no_grad():
                                recons = accelerator.unwrap_model(self.ema.ema_model).sample(condition_seq).permute(0, 2, 1)
                                
                            self.loss_meter.update(chamfer_distance(recons, training_seq.permute(0, 2, 1))[0])

                            if _>=3:
                                break

                        val_data = {"True":training_seq.permute(0, 2, 1).cpu(), "Predicted":recons.cpu(), "loss":self.loss_meter.avg}
                        self.accelerator.log({"val_loss":self.loss_meter.avg})
                        # self.log_samples(condition_seq.cpu(), training_seq.cpu().permute(0, 2, 1), recons.cpu())  

                        self.loss_meter.reset()
        
                        
                        if self.logger == "wandb":
                     
                            for i in range(len(recons)):

                                point_scene = wandb.Object3D({
                                    "type": "lidar/beta",
                                    "points": training_seq.cpu().permute(0, 2, 1)[i].numpy(), 

                                    })
                                self.accelerator.log({f"True point_cloud{i+1}": point_scene, f"Reconstructed point_cloud{i+1}": wandb.Object3D(recons.cpu()[i].numpy())},)
                                if i==7:    
                                    break
                            images_t = wandb.log({"Measurement": [wandb.Image(im) for im in condition_seq[:8]]})
                                
                        self.save(milestone, val_data=val_data)

                self.step += 1
                pbar.update(1)

        accelerator.print('training complete')


if __name__ == "__main__":
    
    # Datasets and loaders
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--path_to_data', type=str, default="/home/predstan/research/mitsuba/tmp",)
    parser.add_argument('--loss_type', type=str, default="l2",)
    parser.add_argument('--objective', type=str, default="pred_noise")
    parser.add_argument('--learned_variance', type=int, default=0)                        
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=12)
    parser.add_argument('--val_freq', type=int, default=20000)
    
    parser.add_argument('--results_folder', type=str, default="./checkpoints/SSD")
    parser.add_argument('--experiment_name', type=str, default="SSD")
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--timesteps', type=int, default=512)
    parser.add_argument('--image_channel', type=int, default=3)
    parser.add_argument('--seq_length', type=int, default=2048)
    
    def seed_all(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    args = parser.parse_args()
    seed_all(args.seed)
    
    Trainer = OccluderTrainer(
        path_to_data="./data//Objects3", 
        args=args,
        experiment_name=args.experiment_name,
        train_batch_size = args.train_batch_size,
        val_batch_size = args.val_batch_size,
        gradient_accumulate_every =1,
        train_lr = 1e-4,
        train_num_steps = 5000000,
        ema_update_every = 10,
        train_maximum=-100,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = args.val_freq,
        num_samples = 25,
        results_folder = args.results_folder,
        amp = False,
        fp16 = False,
        split_batches = True,
        logger = "wandb")
    
    if args.resume:
        print("Resuming from Trained Diffusion")
        Trainer.load(int(args.resume.split("-")[-1].split(".")[0]))
        # Trainer.step = data["step"]

    Trainer.train()

