from occluder.models.ssd import SSD
from occluder.utils.utils import *
from occluder.forward_model.world_ import *
from occluder.forward_model.fista import *
# from image_utils import *
from Trainer import *
import trimesh
from sdf_model.implicit.caller import SDFModel
import numpy as np
from torchvision.ops import nms


def point_as_occluder(point0, min_box=None, max_box=None, x=(0.55, 0.7), y=(0.6, 0.7), z=(0.15, 0.35), flip=False):
    x_min, x_max =x; y_min, y_max =y ; z_min, z_max = z
    point0 = torch.Tensor(point0).clone()
    reshape = len(point0.shape) < 3
    if reshape:
        point0 = point0.unsqueeze(0)
    if flip:
        point0 = point0[:, :,  [0, 2, 1]].clone()
        
    minimum = point0.min(1)[0].unsqueeze(1)
    maximum = point0.max(1)[0].unsqueeze(1)
    if min_box is None:
        val_min = min_box
        val_max = max_box
        
    else:
        val_min = min_box.unsqueeze(0).to(point0)
        val_max = max_box.unsqueeze(0).to(point0)
        
    # print(point0.shape, minimum.shape, maximum.shape, val_min.shape, val_max.shape)

    point0 = ((point0 - minimum)/(maximum-minimum))*(val_max-val_min) + val_min

    if reshape:
        return point0[0]

    return point0



def predict_location(B, m, possible_locations, mesh, optimizer="cgd"):
    possible_locations = list(reversed(possible_locations))
    original_verices = torch.from_numpy(mesh.vertices).type(torch.float32).to(m.device)
    ms=[]; occluders=[]
    with torch.no_grad():
        minn, idx = 987, 0
        mdd, m22, sc = 0, 0, 0
        b = 0
        j = 0
        
        for i in range(len(possible_locations)):

            box = np.array(possible_locations[i])
            box =  torch.from_numpy(box).type(torch.float32).to(m.device)
           
            mesh.vertices = point_as_occluder(original_verices, box[:3], box[3:]).cpu().numpy()
            md = B.GetForwardModelFromMesh(mesh) 
            

            if optimizer=="cgd":
                sc1 = B.GetCGD(m, n_iteration=10, Model=md)

            else:
                sc1 = B.GetTikRegularizedScene(m, lambda_reg=0.1, Model=md)
      
            m2 = B.get_measurement(sc1, Model=md)
            loss = torch.nn.functional.mse_loss(m/m.max(), m2/m2.max())
            
            if loss.item()<minn:
                minn = loss.item()
                mdd = md #.cpu()
                m22 = m2 #.cpu()
                b= box #.cpu()
                sc =sc1 #.cpu()

                ms.append(m2)
                occluders.append((np.asarray(mesh.vertices), np.asarray(mesh.faces)))
                print(i)
            j+=1
            print(i, minn)

      
 
        return mdd.type(torch.float32), m22.type(torch.float32), sc.type(torch.float32), b.type(torch.float32), ms, occluders
    


def estimate(measurements, light_transport=None, occluder_relative_size=(.1, .1, .1), resolution=64, seed=2023):

    np.random.seed(seed)
    torch.manual_seed(seed)


    device = dev="cuda:0"
    M = (128, 128); N = (resolution, resolution)

    def get_ForwardModel(N, M, device):
        

        grid_size =  [3, 3, 3] #Define Occluder Discritization Not Used in SSD
        B = PinsPeck(hidden_x_range=[0, .408], hidden_y_range=(0.0, 0.0), hidden_z_range=[0.03, 0.03+0.3085],
                    visible_x_range=[0.218, 0.218+0.9], visible_y_range=(1.076, 1.0), visible_z_range=[0.05, 0.05+0.9],
                    hidden_resolution=N[0], visible_resolution=M[0], sub_resolution=1, device=device, grid_size=grid_size)

        return B

    if light_transport is None:
            
        B = get_ForwardModel(N, M, device)


    if isinstance(measurements, dict):
        measurement = torch.load(measurements["measurement"])
        occluder_relative_size = measurements["occluder_size"]

    else:
        measurement=measurements


    measurement=measurement/measurement.max()


    sdfModel = SDFModel(device=device, ckpt="./checkpoints/sdf.pt")
    ckpt = torch.load("./checkpoints/ssd.pt", map_location="cpu")
    args= ckpt["args"]
    diffusion= SSD(3, args.latent_dim, learned_variance=args.learned_variance,
                    seq_length=args.seq_length, device=device, image_channel=args.image_channel, timesteps=args.timesteps)
            
    diffusion.to(device)
    diffusion.eval()
    diffusion.load_state_dict(ckpt["model"])


    x = diffusion.sample(torch.flip(measurement.to(device).permute(0, 3, 1, 2), (2,)), gudiance_scale=0.0,sample_time= 256, seq_length=15096, return_all=True).cpu().permute(0, 2, 1)
    vs = point_as_occluder(x[0], torch.tensor([-0.4, -0.4, -0.4]), torch.tensor([0.4, 0.4, 0.4]))
    
    mesh = sdfModel.mesh_from_pointcloud(vs)[0]


    _, measured, scene_, found_box, ms, occluders = predict_location(B, measurement.to(device), torch.from_numpy(np.array(B.generate_sweeping_bounding_boxes(*occluder_relative_size, 5,))).type(torch.float32), mesh)



    original_vertices = torch.from_numpy(mesh.vertices).type(torch.float32).to(device)
    mesh.vertices = point_as_occluder(original_vertices, found_box[:3], found_box[3:]).cpu().numpy()
    md = B.GetForwardModelFromMesh(mesh) 
    mode = md
    r = reconstructTV(B, measurement[:, :, :, 0:1], x_initial=None, mode=mode, num_iteration=500, lamda=1e-3, prox_Lips=None, proximal_iter=400, eps=1e-3)
    g = reconstructTV(B, measurement[:, :, :, 1:2], x_initial=None, mode=mode, num_iteration=500, lamda=1e-3, prox_Lips=None, proximal_iter=400, eps=1e-3)
    b = reconstructTV(B, measurement[:, :, :, 2:], x_initial=None, mode=mode, num_iteration=500, lamda=1e-3, prox_Lips=None, proximal_iter=400, eps=1e-3)


    scene_ = torch.concat([r,g,b], -1).cpu()
    scene_= (scene_[0].cpu()/scene_.cpu().max())*1.2

    # Example usage
    pc = point_as_occluder(x[0], torch.tensor([0, 0,0]), torch.tensor([1, 1, 1]))
    if isinstance(measurements, dict):
        measurements["pointcloud"] = pc.numpy()
        measurements["mesh"] = (np.asarray(mesh.vertices), np.asarray(mesh.faces))
        measurements["scene"] = scene_.cpu().numpy()
        return measurements, ms, occluders
    





scenes = {"ball_smiles":{"measurement":"measurements/ball_on_smile.pt",
                         "occluder_size":(0.1, 0.1, .1)}, 
           "ball_complex":{"measurement":"measurements/ball_on_complex.pt",
                           "occluder_size":(0.1, 0.1, 0.1,)},
            "real_chair":{"measurement":"measurements/random_real_ball_on_chair.pt",
                          "occluder_size":(0.15, 0.02, 0.22,)},
            "random_mush": {"measurement":"measurements/random_on_mush.pt", 
                            "occluder_size":(0.15, 0.02, 0.22,)} }



# Reconstructed all Scenes
for scene in scenes:
    reconstructed = estimate(measurements=scenes[scene])
    scenes[scene] = reconstructed

# Reconstructed a single Scenes
# reconstructed, projected_measurement, occluders = estimate(measurements=scenes["real_chair"])

