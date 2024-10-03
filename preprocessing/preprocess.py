from occluder.forward_model.world_model import *
from occluder.forward_model.world_ import *
import glob
import trimesh
import os
from torchvision import transforms as T, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from occluder.datasets.pointcloud import PointCloud
import scipy
import math
# import objaverse





# objaverse._VERSIONED_PATH = os.path.join("/data/fraji/.objavers/", "hf-objaverse-v1")
# objaverse.BASE_PATH

def rotate_pointcloud(pointcloud, degrees, axis):
    """
    Rotate a point cloud around a specific axis by a given number of degrees using PyTorch.
    
    Args:
        pointcloud (torch.Tensor): Input point cloud of shape (N, 3), where N is the number of points.
        degrees (float): Number of degrees to rotate the point cloud.
        axis (int): Axis to rotate around (0 for X-axis, 1 for Y-axis, 2 for Z-axis).
    
    Returns:
        torch.Tensor: Rotated point cloud.
    """
    radians = math.radians(degrees)
    rotation_matrix = torch.eye(3)
    
    cos_theta = torch.cos(torch.tensor(radians))
    sin_theta = torch.sin(torch.tensor(radians))
    
    if axis == 0:
        rotation_matrix[1, 1] = cos_theta
        rotation_matrix[1, 2] = -sin_theta
        rotation_matrix[2, 1] = sin_theta
        rotation_matrix[2, 2] = cos_theta
    elif axis == 1:
        rotation_matrix[0, 0] = cos_theta
        rotation_matrix[0, 2] = sin_theta
        rotation_matrix[2, 0] = -sin_theta
        rotation_matrix[2, 2] = cos_theta
    elif axis == 2:
        rotation_matrix[0, 0] = cos_theta
        rotation_matrix[0, 1] = -sin_theta
        rotation_matrix[1, 0] = sin_theta
        rotation_matrix[1, 1] = cos_theta
    else:
        raise ValueError("Invalid axis. Must be 0, 1, or 2.")
    
    rotated_pointcloud = torch.matmul(pointcloud, rotation_matrix)
    return rotated_pointcloud



def point_as_occluder(point0, bounding_box_min, bounding_box_max):
       
       
    # Compute the bounding box of the mesh
    point0 = torch.from_numpy(point0)
    mesh_min = point0.min(0)[0] #torch.min(point0, axis=0)
    mesh_max = point0.max(0)[0] #(point0, axis=0)
   
    mesh_dims = mesh_max - mesh_min
    bounding_box_dims = bounding_box_max - bounding_box_min
   
    longest_dim_index = torch.argmax(mesh_dims)
   
    ratio = mesh_dims/mesh_dims[longest_dim_index]
    ratio[ratio<0.7] = 0.7
    rat = bounding_box_dims[longest_dim_index]*ratio

    point0 = ((point0 - mesh_min)/(mesh_max-mesh_min))*(rat) + bounding_box_min

    return point0


class Dataset(Dataset):
    def __init__(
        self,
        path_to_img = "/home/predstan/research/imagenet/imagenet_images",
        image_size = 128,
        index=0,
    ):
        super().__init__()
        self.folder = path_to_img
        self.image_size = image_size
        
        if not paths_global:
            self.paths = glob.glob(path_to_img+"/**/**.jpg") #[p for ext in exts for p in Path(f'{path_to_img}').glob(f'**/*.{ext}')]
            
        else:
        
            if index is not None:
                overall = len(paths_global)//len(files) if len(files)<len(paths_global) else 1
                self.paths = paths_global[overall*index : overall*index+overall]


            # print(index, overall*index, overall*index+overall)
            
        # random.shuffle(self.paths)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert("RGB")
        img = self.transform(img)


        return img, "/".join(path.split("/")[-3:])



def torch_dataset(path="/home/predstan/imagenet/imagenet_images", 
                    batch_size=32,
                    image_size=128,
                    index=0):
    
    return Dataset(image_size=image_size, path_to_img=path, index=index)
    

def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
          
            mesh = scene_or_mesh.dump(concatenate=True)
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def get_ForwardModel(N, M, device):

    
    grid_size =  [32, 16, 32] #Define Occluder Discritization
    B = PinsPeck(hidden_x_range=[0, .408], hidden_y_range=(0.0, 0.0), hidden_z_range=[0.03, 0.03+0.3085],
                visible_x_range=[0.218, 0.218+0.9], visible_y_range=(1.076, 1.0), visible_z_range=[0.05, 0.05+0.9],
                hidden_resolution=N[0], visible_resolution=M[0], sub_resolution=1, device=device, grid_size=grid_size)
    # B.ComputeMatrix()

    return B


import os
def torchsave(p, parent_path, file):
    isExist = os.path.exists(parent_path)
    
    if not isExist:
        os.makedirs(parent_path)
        print("created = ",parent_path)
    filename = parent_path+file+".pt"
    torch.save(p, filename)
    return True


def matsave(val, parent_path, file):
    isExist = os.path.exists(parent_path)
    # print(parent_path)
    if not isExist:
        os.makedirs(parent_path)
        print("created = ",parent_path)
    filename = parent_path+file+".mat"

    scipy.io.savemat(filename, val)
    # torch.save(p, filename)
    return True


boxes = None
def Grab_process(i, file):
    
    
    global boxes
    

    folder = file.split("/")[4]
    
    # /home/predstan/research/occlude/dataset_5000/measurements/00015.mat
    # print(f"{save_path}+/measurements+/{i+1:05d}.mat")
    if os.path.isfile(f"{save_path}/Objects3/{i+1:06d}.mat"):
        print(i, "done")
        return
 

    model = A[device_dict[i]]
    
    
    if boxes is None:
        boxes = np.array(model.generate_sweeping_bounding_boxes(0.2, 0.2, 0.2, 100))
        # boxes = np.concatenate([boxes, model.generate_sweeping_bounding_boxes(0.15, 0.15, 0.15, 50)])
        # boxes = np.concatenate([boxes, model.generate_sweeping_bounding_boxes(0.17, 0.17, 0.17, 10)])
        boxes = np.concatenate([boxes, model.generate_sweeping_bounding_boxes(0.25, 0.25, 0.25, 200)]) #np.array(model.PointDiv(ranges=40)) # np.concatenate([np.array(model.PointDiv(x_diff=0.3, y_diff=0.3, z_diff=0.3, ranges=20))])
    

    try:

        data = torch_dataset(path="/data/fraji/tmp", batch_size=repeated_m, index=i,
                                    image_size=N[0])
        

        scene, image_path = data[0]
        scene = scene.permute(1, 2, 0)[None]

    except:
        return

    
    box = boxes[np.random.randint(0, len(boxes)) ]
    mesh = trimesh.load(file, force='mesh')

    if np.random.choice([0, 1], p=[0.0, 1.]):
        
        mesh.vertices = point_as_occluder(mesh.vertices[:, [0,2,1]] , box[:3], box[3:]) #.numpy()
        
    else:
        
        mesh.vertices = point_as_occluder(mesh.vertices, box[:3], box[3:])
        
    r = np.random.choice([0, 1, 2], p=[0.9, 0.05, 0.05])
    
    if r:
        r = 180 if r == 1 else np.random.choice(range(180))
        v = rotate_pointcloud(torch.from_numpy(mesh.vertices.astype(np.float32)), r, 2).numpy()
        mesh.vertices = point_as_occluder(v, box[:3], box[3:])


    mode = model.GetForwardModelFromMesh(mesh, 64, 0)
    m = model.get_measurement(scene, Model=mode).cpu().numpy()[0]


    file = "/".join(file.split("/")[4:])

    v, sdf = trimesh.sample.sample_surface(mesh, 50000)

    pc = PointCloud(coords=v, channels={})
    pc = pc.farthest_point_sample(2048)
    v = pc.coords.astype(np.float32)

    
    val = {"measurement":m, "scene":image_path, "model_path":file, "box":box, "rotated": r}
    val["pointcloud"] = v
    matsave(val, save_path+"/Objects3", f'/{i+1:06d}')
    print("completed", i)
  


import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ ==  "__main__":


    import argparse
    import random
    


    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_img', type=str, default="/mitsuba/tmp")
    parser.add_argument('--image_type', type=str, default="png")
    parser.add_argument('--save_path', type=str, default='./')


    parser.add_argument('--seed', type=int, default=2023)

    def seed_all(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)



    args = parser.parse_args()
    seed_all(args.seed)

    save_path = args.save_path


    from joblib import Parallel, delayed
    import random
    # ShapeNetCore.v2
    files = glob.glob("./ShapeNetCore_no_textures.v2/**/**/models/**.obj")*5#[13700:]#*4

    print("Completed Loading")
    print(len(files))


    repeated_m = 1
    M = (128, 128)
    N = (32, 32)

    ngpu = torch.cuda.device_count()
    start_gpu = 0

    import pynvml
    path_to_img = args.path_to_img
    if path_to_img is not None:
        paths_global = glob.glob("./mitsuba/tmp/**/**.png")
        random.shuffle(paths_global)
    saves = list(range(len(files)))
    print(len(paths_global))

    
    def get_memory_free_MiB(gpu_index):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.free // 1024 ** 3

    print("Proceeded")
    single = get_memory_free_MiB(0) // 5
    process = []
    hold = {}

    for j in range(single):
        for i in range(start_gpu, ngpu):
            process.append(i)
                  
    # print(single, process, len(process), ngpu)
    device_dict = {}
    j=0
    for i in range(0, len(files)):
        if j==len(process)-1:
            j=0
        device_dict[i] = j
        j+=1


    A = {}
    for i in range(len(process)):

        device = "cuda:"+str(process[i])
        # print(process[i])
    
        model = get_ForwardModel(N, M, device)
        A[i] = model
    
    print(f"starting {len(process)} threads")
    # Grab_process(0, files[0])
    # for i, file in enumerate(files):
    #     Grab_process(i+13617, files[0])
    Parallel(n_jobs=len(process), prefer="threads")(delayed(Grab_process)(i, file) for i, file in enumerate(files))

      