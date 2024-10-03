# from torch._C import device
from implicit.sdf_vaes import *
import skimage
import trimesh
import scipy
from einops import rearrange, reduce


class Query(object):
    
    
    def __init__(self, grid_size=128, side_length=1.02, device="cpu", occluder_space=None) -> None:

        self.voxel_size = side_length / (grid_size - 1)
        self.min_coord = -side_length / 2
        self.grid_size = grid_size
        self.side_length=side_length
        self.device = device
        self.indices = torch.tensor(range(0, self.grid_size**3), device=self.device)
        self.occluder_space = torch.tensor(list(occluder_space)) if occluder_space is not None else None
            

    def int_coord_to_float(self, int_coords: torch.Tensor) -> torch.Tensor:
        return int_coords.float() * self.voxel_size + self.min_coord
    
    
    def sample(self, batch_size, num_points):
        ind = torch.randint(0, len(self.indices), size=(batch_size, num_points), device=self.device)
        indices = self.indices[ind]
        zs = self.int_coord_to_float(indices % self.grid_size)
        ys = self.int_coord_to_float(torch.div(indices, self.grid_size, rounding_mode="trunc") % self.grid_size)
        xs = self.int_coord_to_float(torch.div(indices, self.grid_size**2, rounding_mode="trunc"))
        coords = torch.stack([xs, ys, zs], dim=0).permute(1, 2, 0)
        return coords, indices
    
    
    def __getitem__ (self, index):
        indices = self.indices[index]
        zs = self.int_coord_to_float(indices % self.grid_size)
        ys = self.int_coord_to_float(torch.div(indices, self.grid_size, rounding_mode="trunc") % self.grid_size)
        xs = self.int_coord_to_float(torch.div(indices, self.grid_size**2, rounding_mode="trunc"))
        coords = torch.stack([xs, ys, zs], dim=-1).type(torch.float32)
        return coords
    
    def get_all(self):
        zs = self.int_coord_to_float(self.indices % self.grid_size)
        ys = self.int_coord_to_float(torch.div(self.indices, self.grid_size, rounding_mode="trunc") % self.grid_size)
        xs = self.int_coord_to_float(torch.div(self.indices, self.grid_size**2, rounding_mode="trunc"))
        coords = torch.stack([xs, ys, zs], dim=-1).type(torch.float32)
        return coords
    
    
    def sample_outside_cube(self, num_points, outer_bound=1.0):
        
        points = []
        for _ in range(3):
            rand = torch.randint(0, 2, (1,))
            range_start = -outer_bound if rand>=1 else self.side_length / 2
            range_end = -self.side_length / 2 if rand>=1 else outer_bound
            range_width = range_end - range_start

            # Generate random points
            points.append(torch.rand(num_points, device=self.device) * range_width + range_start)
            
        grid_x, grid_y, grid_z = torch.meshgrid(points[0], points[1], points[2], indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
        return coords
    
    
    def renormalize_occluder(self, x:torch.Tensor, norm=False):
        ndim = x.ndim
        assert ndim in (4, 3, 2) and self.occluder_space is not None
        occ = self.occluder_space[None].to(x) if ndim == 2 else self.occluder_space[None, None].to(x)
        occ = occ[None] if ndim == 4 else occ
        if ndim != 4:
            min_values, max_values = (occ[:, :3], occ[:, 3:]) if ndim == 2 else (occ[:, :, :3], occ[:, :, 3:])
        else:
            min_values, max_values = (occ[:, :,:, :3], occ[:, :, :, 3:])
        
        normalized = (x - min_values) / (max_values - min_values) if not norm else x
        return normalized * self.side_length + self.min_coord
        
    
    def __len__ (self):
        return len(self.indices)
    
    # /home/predstan/research/SSD/checkpoints/sdf-10.pt

class SDFModel(nn.Module):
    
    def __init__(self, device, ckpt="./../../checkpoints/model-20.pt", resolution=128):
        super().__init__()

        ckpt = torch.load(ckpt, map_location="cpu")
        
        self.model = VAEPointCloudSDFModel(device=device, vae=ckpt["args"].vae, with_ae=ckpt["args"].with_ae).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.device=device

        self.query = Query(grid_size=resolution, device=device, side_length=1.02)


    def encode(self, pointcloud):
        if len(pointcloud.shape) == 2:
            pointcloud = pointcloud[None]
        return self.model.encode_point_clouds(pointcloud.to(self.device)).detach()
    

    def mesh_from_latent(self, z):
        return self(z)
    
    def mesh_from_pointcloud(self, pointcloud):

        
        if len(pointcloud.shape) == 2:
            pointcloud = pointcloud[None]

        with torch.no_grad():
            z = self.model.encode_point_clouds(pointcloud.to(self.device)).detach()
        return self(z)
        

    def forward(self, z):


        with torch.no_grad():
    
            sdfs = []
            for i in range(0, len(self.query), 80000):
                xyz = torch.stack([self.query[i:i+80000]]*len(z))
                sdfs.append(self.model.predict_occupancy( xyz, z))

            sdfs = torch.cat(sdfs, 1)
            meshes = []
            for sdf in sdfs:
                sdf1 = sdf.view(self.query.grid_size, self.query.grid_size, self.query.grid_size).cpu().numpy()
                verts, faces, normals, _ = skimage.measure.marching_cubes(
                                                    volume=sdf1,
                                                    level=0,
                                                    allow_degenerate=False,
                                                    spacing=(self.query.voxel_size,) * 3,
                                                    )
                verts+= self.query.min_coord
                
                mesh = trimesh.Trimesh(
                                vertices=verts,
                                faces=faces,
                                normals=normals,
                            )
                meshes.append(mesh)
                
            return meshes
        
def point_as_occluder(point0, bounding_box_min, bounding_box_max):
       
       
    # Compute the bounding box of the mesh
    mesh_min = point0.min(0)[0] #torch.min(point0, axis=0)
    mesh_max = point0.max(0)[0] #(point0, axis=0)
   
    mesh_dims = mesh_max - mesh_min
    bounding_box_dims = bounding_box_max - bounding_box_min
   
    longest_dim_index = torch.argmax(mesh_dims)
   
    ratio = mesh_dims/mesh_dims[longest_dim_index]
    # ratio[ratio<0.3] = 0.3
    rat = bounding_box_dims[longest_dim_index]*ratio
    point0 = ((point0 - mesh_min)/(mesh_max-mesh_min))*(rat) + bounding_box_min
    return point0


# import trimesh
# x= torch.load("/home/predstan/research/SSD/torch_object/test_1.pt")
# x = torch.tensor(x[321]["pointcloud"])
# print(x.shape)

# device="cuda"
# model = SDFModel(device)
# x= point_as_occluder(x,torch.tensor([-.3, -.3, -.3]), torch.tensor([.3, .3, .3]))
# x=model.encode(x.to(device))
# print(x.max(), x.min())
# x=model.mesh_from_latent(x)[0]
# predicted_occluder, sdf = trimesh.sample.sample_surface(x, 40096)
# x = model.mesh_from_pointcloud(torch.Tensor(np.asarray(predicted_occluder)))[0]
# x.show()
# print(x.max(), x.min())