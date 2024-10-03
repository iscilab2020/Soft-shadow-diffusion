
import torch
from torch import einsum
from einops import rearrange
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Function
import pickle
import trimesh

def line_intersection_point(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines does not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
   

def occ_range(D, y, xs, xm):
    x_min = (y[1]/D)*(xm[0] - xs[0]) + xs[0]
    x_max = (y[0]/D)*(xm[1] - xs[1]) + xs[1]
    return (x_min, x_max)



def find_intercepting_line(scene, cams, depth):

    D = depth
    x0, x1 = scene[0], cams[-1]
    y0, y1 = 0, D
    line1 = [[x0, y0], [x1, y1]]


    x0, x1 = scene[1], cams[0]
    y0, y1 = 0, D
    line2 = [[x0, y0], [x1, y1]]

    return line1, line2

def find_parallel_line(scene, cams, depth):

    D = depth
    x0, x1 = scene[0], cams[0]
    y0, y1 = 0, D
    line1 = [[x0, y0], [x1, y1]]


    x0, x1 = scene[1], cams[1]
    y0, y1 = 0, D
    line2 = [[x0, y0], [x1, y1]]

    return line1, line2



def awgn(s, SNR_min=100, SNR_max=None, L=1, return_snr=False):
    shape = s.shape
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    s = torch.Tensor(s)
    device = s.device

    assert len(shape) == 4, f"Expect 4 Dim, Got {shape} Dim to use"

    if SNR_max:
        SNRdB = torch.randint(low=SNR_min, high=SNR_max, size=(shape[0],)).to(device)
    else:
        if isinstance(SNR_min, int):
            return_snr = False

            SNRdB = torch.ones((shape[0],)).to(device)*SNR_min
        else:
            SNRdB = torch.tensor(SNR_min).to(device)

    s = torch.reshape(s, [s.shape[0], -1])
    gamma = 10**(SNRdB/10)


    P = L *torch.sum(torch.abs(s)**2, dim=1)/s.shape[-1]
    N0 = P/gamma
    n = torch.sqrt(N0/2).unsqueeze(1)*torch.rand(s.shape).to(device)
    s = s+n
    if return_snr:
        return torch.reshape(s, shape), SNRdB
    else:
        return torch.reshape(s, shape)
    


def sbr(s, SNR_min=100, SNR_max=None, L=1, return_snr=False, background=None):
    shape = s.shape
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    s = torch.Tensor(s)
    device = s.device
    
   

    assert len(shape) == 4, f"Expect 4 Dim, Got {shape} Dim to use"

    if SNR_max:
        SNRdB = torch.randint(low=SNR_min, high=SNR_max, size=(shape[0],)).to(device)
    else:
        if isinstance(SNR_min, int):
            return_snr = False

            SNRdB = torch.ones((shape[0],)).to(device)*SNR_min
        else:
            SNRdB = torch.tensor(SNR_min).to(device)

    s = torch.reshape(s, [s.shape[0], -1])
    background =  torch.ones_like(s) if not background else (background/background.max()).reshape(1, -1)
    gamma = 10**(SNRdB/10)
    # Calculate signal power
    P = torch.sum(torch.abs(s)**2, dim=1) / s.shape[-1]
    # Calculate background power based on desired SBR
    B0 = P / gamma
    # Calculate the scaling factor of the background based on desired SBR
    scale_factor =  torch.sqrt(B0/(torch.sum(torch.abs(background)**2, dim=1) / background.shape[-1]))
    # Generate background signal and scale it
    b = scale_factor.unsqueeze(1) * background
    # Add background to the signal
    s = s + b
    if return_snr:
        return torch.reshape(s, shape), SNRdB
    else:
        return torch.reshape(s, shape)


class NLOS:
    def __init__(self, hidden_x_range,  hidden_z_range,
                 visible_x_range, visible_y_range, visible_z_range, hidden_y_range=0,
                 visible_resolution=10, hidden_resolution=10, sub_resolution=3,
                 device ="cpu"):
       
       
        self.default_dtype = torch.float32
       
        self.hidden_x_range = hidden_x_range
        self.hidden_z_range = hidden_z_range
       
        self.device = device
       
        self.visible_x_range = visible_x_range
        self.visible_z_range = visible_z_range
        self.visible_y_range = visible_y_range
       
        self.hidden_resolution = hidden_resolution
        self.visible_resolution = visible_resolution 
       
       
               
        self.hidden_pixel_size_x = (hidden_x_range[1] - hidden_x_range[0]) / hidden_resolution # Ger pixels for the x range
        self.hidden_pixel_size_z = (hidden_z_range[1] - hidden_z_range[0]) / hidden_resolution 
       

        # Discretize the space for visible plane using torch meshgrid with specified resolution
        self.hidden_z, self.hidden_x= torch.meshgrid(torch.linspace(hidden_z_range[0]+(0.5*self.hidden_pixel_size_z),
                                                    hidden_z_range[1]-(0.5*self.hidden_pixel_size_z), hidden_resolution, device=device),
                                            torch.linspace(hidden_x_range[0]+(0.5*self.hidden_pixel_size_x),
                                                    hidden_x_range[1]-(0.5*self.hidden_pixel_size_x), hidden_resolution, device=device),
                                            indexing="ij")
       
       
       
        hidden_y = torch.ones_like(self.hidden_x, device=device)*hidden_y_range[0]  # Ensure y-coordinate is 0 for the hidden scene
        self.hidden = self.cast(torch.stack((self.hidden_x.t().flatten(), hidden_y.flatten(), self.hidden_z.t().flatten()), dim=-1))
       
       
       
        self.visible_pixel_size_x = (visible_x_range[1] - visible_x_range[0]) / visible_resolution # Ger pixels for the x range
        self.visible_pixel_size_z = (visible_z_range[1] - visible_z_range[0]) / visible_resolution 
       

        # Discretize the space for visible plane using torch meshgrid with specified resolution
        self.visible_z, self.visible_x = torch.meshgrid(torch.linspace(visible_z_range[0]+(0.5*self.visible_pixel_size_z),
                                                             visible_z_range[1]-(0.5*self.visible_pixel_size_z), visible_resolution, device=device),
                                              torch.linspace(visible_x_range[0]+(0.5*self.visible_pixel_size_x),
                                                             visible_x_range[1]-(0.5*self.visible_pixel_size_x), visible_resolution, device=device),
                                              indexing="ij")
       
       
        visible_y = torch.ones_like(self.visible_x, device=device) * visible_y_range[0]  
        self.visible = self.cast(torch.stack((self.visible_x.t().flatten(), visible_y.t().flatten(), self.visible_z.t().flatten()), dim=-1))
       
       
        # print(self.visible)
        # print(self.hidden)
        self.sub_resolution = sub_resolution
       
        # Compute the intensity matrix
        self.Model = self.compute_intensities(sub_resolution)
       
       
    def cast(self, inputs, type=None):
        """
        Helper Function for Casting Array into Defined Dtype
        """
        if type is None:
            type = self.default_dtype

        if torch.is_tensor(inputs):
            inputs = inputs.type(type)
        else:
            inputs = torch.as_tensor(inputs, dtype=type)

        if inputs.device != self.device:
            return inputs.to(self.device)

        return inputs
       
       
    def compute_intensities(self, sub_resolution, batch_size=32):
        # Initialize the final intensity tensor
        Model = torch.zeros(self.visible.shape[0], self.hidden.shape[0], device=self.hidden.device)
       
        self.update_sub_pixels(sub_resolution)
       
        # Compute intensities in batches to manage memory usage
        for i in range(0, self.hidden.shape[0], batch_size):
            batch_end = min(i + batch_size, self.hidden.shape[0])
            batch_hidden = self.hidden[i:batch_end]
           
            # Compute intensities for the current batch
            batch_intensities = self.compute_sub_pixel_intensities(batch_hidden)
           
            # Store the computed intensities in the final tensor
            Model[:, i:batch_end] = batch_intensities
       
        return Model
   
    def update_sub_pixels(self, sub_resolution):

        step_size_x = self.hidden_pixel_size_x / (sub_resolution+1)
        step_size_z = self.hidden_pixel_size_z / (sub_resolution+1)
       
        step_x = torch.arange(self.hidden_x_range[0]+step_size_x, self.hidden_x_range[0]+self.hidden_pixel_size_x-(step_size_x/2), step_size_x, device=self.device)-self.hidden[0, 0]
        step_z = torch.arange(self.hidden_z_range[0]+step_size_z, self.hidden_z_range[0]+self.hidden_pixel_size_z-(step_size_z/2), step_size_z, device=self.device)-self.hidden[0, -1]

 
        z, x = torch.meshgrid(step_z, step_x, indexing="ij")
       
        sub_pixel_offsets = torch.stack((x.t().flatten(), torch.zeros_like(z.flatten()), z.t().flatten()), dim=-1)
        self.hidden = self.hidden[:, None] + sub_pixel_offsets[None]
   
   
    def compute_sub_pixel_intensities(self, batch_hidden):
       
        shape = batch_hidden.shape[:-1]
        x = batch_hidden.reshape(-1, 3)
        p_x = x[:, None] - self.visible
        numerator = p_x[:, :, 1]*p_x[:, :, 1]
       
        intensity = numerator[:, :,]/(p_x**4).sum(-1)
       
   
        return intensity.reshape(*shape, -1).mean(1).t()
            # return intensity



    def GetTikRegularizedScene(self, measurement:np.ndarray, lambda_reg: float = 0.001, Model=None, shape=None) -> np.ndarray:
       
        shape = (self.hidden_resolution, self.hidden_resolution) if shape is None else shape
       
        if Model is None:
            Model = self.Model

        transpose = Model.t()
       
        reg_init = torch.einsum('bj, jk -> bk', transpose, Model)
        I = lambda_reg * torch.eye(transpose.shape[0]).to(Model)


        linv = torch.inverse(reg_init + I)
       
        m = rearrange(measurement, 'b w h c -> b  (h w) c')
        p = torch.matmul(self.cast(linv), torch.matmul(transpose, m))
       
        return rearrange(p, 'b (h w) c -> b w h c', w=shape[0], h=shape[1])
   

    def GetVisibilityFromBox(self, box):


        verts = self.GetOccluderMesh([box])
        mode = self.GetForwardModelFromMesh(verts, return_vis=True)

        return mode
        
        N  = self.hidden.shape[0]

        x, y, z, x1, y1, z1 = tuple(box)

        D  = self.visible_y_range[0]
        shape = self.visible.shape
        campoints_x = self.visible_x.t().flatten()[None].expand(N, shape[0])
        campoints_z = self.visible_z.t().flatten()[None].expand(N, shape[0])



        scenepoints_x = self.hidden[:, 0, 0].unsqueeze(1)
        scenepoints_z = self.hidden[:, 0, 2].unsqueeze(1)


        left_edge = (D * torch.divide(x - scenepoints_x, y)) +  scenepoints_x
        right_edge = (D * torch.divide(x1 - scenepoints_x, y)) +  scenepoints_x

        xss = (torch.where(torch.logical_and(campoints_x >= left_edge, campoints_x <= right_edge), 0., 1.)).unsqueeze(1)


        left_edge = (D * torch.divide(z - scenepoints_z, y)) +  scenepoints_z
        right_edge = (D * torch.divide(z1 - scenepoints_z, y)) +  scenepoints_z

        zss = (torch.where(torch.logical_and(campoints_z >= left_edge, campoints_z <= right_edge), 0., 1.)).unsqueeze(1)

        VisMat = 1 - torch.multiply(torch.permute(1-zss, [0, 2, 1]), (1 - xss))
        return self.cast(torch.reshape(torch.permute(VisMat, [2, 1, 0]), [-1, N]))

   
   
    def GetTranspose(self, measurements, Model=None, type=None, shape=None, reshape=True):
       
        shape = (self.hidden_resolution, self.hidden_resolution) if shape is None else shape
   
        assert len(measurements.shape) == 4, f"Expect 4 Dim, Got {measurements.shape} Dim"

        if Model is None:
            Model = self.Model

        measurements = self.cast(measurements, type)
       
       
        m = rearrange(measurements, 'b w h c -> b  (h w) c')
        p = torch.tensordot(Model, m, dims=((0,), (1,)))
   
        return rearrange(p, '(h w) b c -> b w h c', w=shape[0], h=shape[1]) if reshape else rearrange(p, 'h b c -> b h c')
           
           
           
    def get_measurement(self, images:torch.Tensor, Model=None) -> torch.Tensor:
        assert len(images.shape) == 4, f"Expected 4 Dimimension, Got {images.shape} Dim"
        # assert images.shape[1] == self.N1 and images.shape[2] == self.N2 , \
        #     f"Input must be only one image and have shape of BatchSize X {self.N1} X {self.N2} X ChannelSize"
       
        images = self.cast(images)

        if Model is None:
            Model = self.Model

        images = rearrange(images, 'b w h c -> b  (h w) c')
        m = torch.tensordot(a=Model.t(), b=images, dims=((0,), (1,)))
       
        return rearrange(m, '(h w) b c -> b w h c', w=self.visible_resolution, h=self.visible_resolution)
   
   
    def GetTranspose(self, measurements, Model=None, type=None, shape=None, reshape=True):
       
        shape = (self.hidden_resolution, self.hidden_resolution) if shape is None else shape
   
        assert len(measurements.shape) == 4, f"Expect 4 Dim, Got {measurements.shape} Dim"

        if Model is None:
            Model = self.Model

        measurements = self.cast(measurements, type)
       
       
        m = rearrange(measurements, 'b w h c -> b  (h w) c')
        p = torch.tensordot(Model, m, dims=((0,), (1,)))
   
        return rearrange(p, '(h w) b c -> b w h c', w=shape[0], h=shape[1]) if reshape else rearrange(p, 'h b c -> b h c')
   
   
    def GetCGD(self, measurements, n_iteration=50, Model=None, shape=None, reshape=True, current_estimate=None):
        measurements = self.cast(measurements, torch.float32)
        shape = (self.hidden_resolution, self.hidden_resolution) if shape is None else shape

        assert len(measurements.shape) == 4, f"Expect 4 Dim, Got {measurements.shape} Dim"

        if Model is None:
            Model = self.Model

        Model = self.cast(Model, torch.float32)

        p = self.GetTranspose(measurements, Model=Model, type=torch.float32, shape=shape, reshape=False)
        ATA = torch.einsum('bj, jk -> bk', Model.t(), Model)

        r = p
       
        # Use the current estimate if provided, otherwise start with a zero vector
        x = self.cast(rearrange(current_estimate, 'b w h c -> b (h w) c') if current_estimate is not None else torch.zeros_like(r))

        for _ in range(n_iteration):
            denum = torch.permute(torch.tensordot(ATA, p, dims=((0,), (1,))), [1, 0, 2])

            alpha = torch.norm(r, dim=1) ** 2 / torch.sum(r * denum, dim=1)
            x = x + (alpha[:, None] * p)
            r_new = r - (alpha[:, None] * denum)
            beta = torch.norm(r_new, dim=1) ** 2 / torch.norm(r, dim=1) ** 2
            p = r_new + (beta[:, None] * p)
            r = r_new

        return rearrange(x, 'b (h w) c -> b w h c', w=shape[0], h=shape[1]) if reshape else x
   
   
    def get_verts(self, box):
           
        x_min, y_min, z_min, x_max, y_max, z_max = box

        box_vertices = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ])

        # define the faces of the box (each one is a triangle)
        box_faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 3, 7],
            [0, 7, 4],
            [1, 2, 6],
            [1, 6, 5],
        ],)
       
        return box_vertices, box_faces



    def GetOccluderMesh(self, occluders):
       
       

        mesh = None
       
        for box in occluders:
           
            verts, faces = self.get_verts(box)
           
            cubes = o3d.geometry.TriangleMesh()
            cubes.vertices = o3d.utility.Vector3dVector(verts)
            cubes.triangles = o3d.utility.Vector3iVector(faces)

            if mesh is None:
                mesh = cubes
            else:
                mesh = mesh + cubes

        return mesh.remove_duplicated_triangles()
 
    def GetForwardModelFromMesh(self, mesh, batch_size=64, pinhole=False, return_vis=False, scenePoints=None, return_location=False, visible=None):


        if "trimesh" in str(type(mesh)):
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
            mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(mesh.vertex_normals)
            mesh = mesh_o3d
       

        if scenePoints is None:
            scenePoints = self.hidden

        if visible is None:
            visible = self.visible
       
       
        batch_size = batch_size if len(scenePoints)>batch_size else len(scenePoints)      
       
        vis = []
        # print(self.hidden.shape)
       
        occluder_scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        cubes = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
       
        cube_id = occluder_scene.add_triangles(cubes)
   
        for i in range(0, len(scenePoints), batch_size):
            d, o = self.hidden[i:i+batch_size, :], visible
            d_shape = None
            if len(d.shape) > 2:
                d_shape = (d.shape[0], d.shape[1])
                d = d.view(-1, 3)
   
            direction = -(o[:, None]-d)
            direction = direction/torch.linalg.norm(direction, dim=-1, keepdim=True)
            o = o.unsqueeze(1).expand( o.shape[0], len(d), o.shape[1])
            rays =  torch.cat([o, direction], -1)
           
            rays = o3d.core.Tensor(self.cast(rays, torch.float32).cpu().numpy(),
                        dtype=o3d.core.Dtype.Float32)

            ans = occluder_scene.cast_rays(rays)
            sug = (ans["t_hit"]-ans["t_hit"]).numpy()
            sug = self.cast(np.nan_to_num(sug, nan=1.0))
           
            if d_shape is not None:
                sug = sug.reshape(-1, d_shape[0], d_shape[1])
                sug = sug.mean(-1)
           
            #Pinhole version
            if pinhole:
                vis.append(1-sug)
            else:
                vis.append(sug)

        if return_vis:
            if return_location:
                vis = torch.cat(vis, -1)
   
                n = torch.nonzero(vis).cpu().numpy()


                return n, vis[n[:, 0], n[:, 1]]
            return torch.cat(vis, -1)
           
        return self.Model*torch.cat(vis, -1).to(self.device)
   
   
    def GetMeshSphere(self, points, radius=0.01,):
       
        def create_sphere_mesh(center, resolution=10):
            """Generate a mesh for a sphere with the given center and radius."""
            # Create a unit sphere and scale & translate it to the desired position
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
            mesh.translate(center)
            return mesh
           
       
        all_spheres = [create_sphere_mesh(point) for point in points]
        combined_mesh = all_spheres[0]
        for sphere in all_spheres[1:]:
            combined_mesh += sphere
           
        return combined_mesh
   
   
    def GetForwardModelFromPoints(self, points, radius=0.01, batch_size=64, pinhole=False, return_vis=False,):
       
       
        combined_mesh = self.GetMeshSphere(points, radius)
           
           
        return self.GetForwardModelFromMesh(combined_mesh, batch_size=batch_size,
                                            pinhole=pinhole, return_vis=return_vis,)
       



    def GetPCLocation(self, point_clouds):
   
        N  = self.hidden.shape[0]
        discritization = self.hidden.shape[1]
        D = self.visible_y_range[0]
        point, cood = point_clouds.shape
       
        p = self.cast(point_clouds)[None].expand(N*discritization, point, cood)

        scenepoints_x = self.hidden[:, :, 0]
        scenepoints_z = self.hidden[:, :, 2]

        scenepoints_x = scenepoints_x.flatten().unsqueeze(1)
        scenepoints_z = scenepoints_z.flatten().unsqueeze(1)

        x_points = (D * torch.div(p[:, :, 0] - scenepoints_x, p[:, :, 1])) +  scenepoints_x
        z_points = (D * torch.div(p[:, :, 2] - scenepoints_z, p[:, :, 1])) +  scenepoints_z
        x_points = x_points.reshape(N, discritization, -1)
        z_points = z_points.reshape(N, discritization, -1)


        return [x_points, z_points]
       
       
class PinsPeck(NLOS):
   
    def __init__(self, hidden_x_range, hidden_z_range, visible_x_range,
                 visible_y_range, visible_z_range, hidden_y_range=0,
                 visible_resolution=10, hidden_resolution=10, sub_resolution=3, device="cpu", grid_size=[10, 10, 10]):
        super().__init__(hidden_x_range, hidden_z_range, visible_x_range, visible_y_range,
                         visible_z_range, hidden_y_range, visible_resolution,
                         hidden_resolution, sub_resolution, device)
       
       
        self.grid_size = grid_size
        self.points = self.cast(self.occluderDiscritize(grid_size=grid_size,))


   
    def generate_sweeping_bounding_boxes(self, x_diff=0.1, y_diff=0.05, z_diff = 0.1, ranges=100, entire_area=False):
       
       
        points = []

        if isinstance(ranges, list):
            y_avg = (self.occ_y_max-self.occ_y_min)/ranges[1]
            x_avg = (self.occ_x_max-self.occ_x_min)/ranges[0]
            z_avg = (self.occ_z_max-self.occ_z_min)/ranges[2]
        
        else:
            y_avg = (self.occ_y_max-self.occ_y_min)/ranges
            x_avg = (self.occ_x_max-self.occ_x_min)/ranges
            z_avg = (self.occ_z_max-self.occ_z_min)/ranges

        if entire_area:
            y_avg = (self.visible_y_range[0])/ranges
            x_avg = (self.visible_x_range[1] - self.hidden_x_range[0])/ranges
            z_avg = (self.visible_z_range[1] - self.hidden_z_range[0])/ranges
       
       
        for i in range(ranges):
           
            y = self.occ_y_min + y_avg*i
           
            while y <= self.occ_y_min:
                y+=0.01
           
            while y+y_diff >= self.occ_y_max:
                y-=0.01
               
            ys = (y, y+y_diff)
           
            x_min = self.occ_x_min
            z_min = self.occ_z_min

            for j in range(ranges):
                x = x_min + x_avg*j
               
                while x <= self.occ_x_min:
                    x+=0.01
               
                while x+x_diff >= self.occ_x_max:
                        x-=0.01
                       
                xs = (x, x+x_diff)

                for k in range(ranges):
                    z = z_min + z_avg*k
                   
                    while z <= self.occ_z_min:
                        z+=0.01
                   
                    while z+z_diff >= self.occ_z_max:
                        z-=0.01
                   
                    zs = (z, z+z_diff)
                   
                    points.append([xs[0], ys[0], zs[0], xs[1], ys[1], zs[1]])
                   
        return points
       
       
       
    def occluderDiscritize(self, grid_size):
       
        points = {}

        (y_min, y_max), _, _ = self.OccMinMax(depth_difference=3)
        self.occ_y_max = y_max
        self.occ_y_min = y_min
       
        self.occ_x_min, self.occ_x_max = occ_range(self.visible_y_range[0], (y_min, y_max), self.hidden_x_range, self.visible_x_range)
        self.occ_z_min, self.occ_z_max = occ_range(self.visible_y_range[0], (y_min, y_max), self.hidden_z_range, self.visible_z_range)
       
        self.occ_x_min+=0.05
        self.occ_x_max-=0.05
        self.occ_z_min+=0.05
        self.occ_z_max-=0.05
       
       
       
        self.grid_x_size = (self.occ_x_max - self.occ_x_min) / (self.grid_size[0])
        self.grid_y_size = (self.occ_y_max - self.occ_y_min) / (self.grid_size[1])
        self.grid_z_size = (self.occ_z_max - self.occ_z_min) / (self.grid_size[2])
       
       
        grid_x = torch.arange(self.occ_x_min, self.occ_x_max, self.grid_x_size,)
        grid_y = torch.arange(self.occ_y_min, self.occ_y_max, self.grid_y_size,)
        grid_z = torch.arange(self.occ_z_min, self.occ_z_max, self.grid_z_size,)
       
       
        x, y, z = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
       
        return torch.stack([x.flatten(), y.flatten(), z.flatten()], -1 )
   

    def shift_point_into_grid(self, pointcloud_batch):
        # Assuming pointcloud_batch is of shape (batch_size, num_points, 3)
        batch_size, num_points, _ = pointcloud_batch.shape

        # Calculate grid cell indices for all points in the batch
        i = ((pointcloud_batch[:, :, 0] - self.occ_x_min) / self.grid_x_size).long()
        j = ((pointcloud_batch[:, :, 1] - self.occ_y_min) / self.grid_y_size).long()
        k = ((pointcloud_batch[:, :, 2] - self.occ_z_min) / self.grid_z_size).long()
       
        # Check which points fall inside the grid or touch the border
        valid_mask = (i >= 0) & (i < self.grid_size[0]) & \
                    (j >= 0) & (j < self.grid_size[1]) & \
                    (k >= 0) & (k < self.grid_size[2])

        # Filter the indices using the valid_mask
        valid_i = i[valid_mask]
        valid_j = j[valid_mask]
        valid_k = k[valid_mask]
        valid_b = valid_mask.nonzero(as_tuple=True)[0]  # Extract batch indices of valid points

        # Initialize volume tensor
        volume = torch.zeros(batch_size, *self.grid_size, dtype=torch.float32)

        # Use advanced indexing to update the volume tensor for all valid points
        volume[valid_b, valid_i, valid_j, valid_k] = 1

        i = ((pointcloud_batch[:, :, 0].mean(-1) - self.occ_x_min) / self.grid_x_size).long()
        j = ((pointcloud_batch[:, :, 1].mean(-1) - self.occ_y_min) / self.grid_y_size).long()
        k = ((pointcloud_batch[:, :, 2].mean(-1) - self.occ_z_min) / self.grid_z_size).long()
        # print(i.shape, j.shape)
        centroid = torch.stack([i/self.grid_size[0], j/self.grid_size[1], k/self.grid_size[2]], -1)

        return volume, centroid
   

    def differentiable_shift_point_into_grid(self, pointcloud_batch):
        batch_size, num_points, _ = pointcloud_batch.shape

       


        def gaussian_influence(distance):      
                print(distance.max(), distance.min())    #std
                return torch.exp(-0.5 * (distance / 0.002) ** 2)

        # Create a grid of cell centers
        # x = torch.linspace(self.occ_x_min, self.occ_x_min + self.grid_size[0] * self.grid_x_size, self.grid_size[0])
        # y = torch.linspace(self.occ_y_min, self.occ_y_min + self.grid_size[1] * self.grid_y_size, self.grid_size[1])
        # z = torch.linspace(self.occ_z_min, self.occ_z_min + self.grid_size[2] * self.grid_z_size, self.grid_size[2])
        # xx, yy, zz = torch.meshgrid(x, y, z, indexing="i,j")


       

   
        # grid_centers = torch.stack((xx, yy, zz), dim=-1)

        grid_centers = self.points + self.cast(torch.from_numpy(np.array([[self.grid_x_size/2, self.grid_y_size/2, self.grid_z_size/2]])))
        grid_centers = grid_centers.view(*self.grid_size, 3)

        # Reshape tensors for broadcasting: expanded dimensions should be inserted at the beginning
        pointcloud_batch_expanded = pointcloud_batch.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        grid_centers_expanded = grid_centers.unsqueeze(0).unsqueeze(1)

        # Compute the distance from each point to each grid cell
        distance_to_grid = torch.norm(self.cast(pointcloud_batch_expanded) - grid_centers_expanded, dim=-1)

        # Compute the influence of each point on each grid cell
        influence = gaussian_influence(distance_to_grid)

        # Sum the influences across the point dimension (2nd dimension)
        volume = influence.sum(dim=1)

        # Normalize volume to [0, 1]
        volume = torch.clamp(volume, 0, 1)
        return volume
   
    def get_transformation_matrix(self):
        # Initialize transformation parameters as learnable parameters
        self.translation = torch.nn.Parameter(torch.zeros(3))  # Initialized to zero translation
        self.scaling = torch.nn.Parameter(torch.ones(3))  # Initialized to no scaling (1, 1, 1)
        self.rotation = torch.nn.Parameter(torch.zeros(3))
            # Ensure all tensors are of type float for matrix operations
        self.scaling = self.scaling.float()
        self.rotation = self.rotation.float()
       
        # Scaling matrix
        S = torch.diag(self.scaling)
       
        # Rotation matrix using ZYX (roll, pitch, yaw) order
        sin, cos = torch.sin(self.rotation), torch.cos(self.rotation)
        R_x = torch.tensor([[1, 0, 0],
                            [0, cos[0], -sin[0]],
                            [0, sin[0], cos[0]]], dtype=torch.float32)
        R_y = torch.tensor([[cos[1], 0, sin[1]],
                            [0, 1, 0],
                            [-sin[1], 0, cos[1]]], dtype=torch.float32)
        R_z = torch.tensor([[cos[2], -sin[2], 0],
                            [sin[2], cos[2], 0],
                            [0, 0, 1]], dtype=torch.float32)
        R = R_z @ R_y @ R_x
       
        # Combine scaling and rotation
        T = R @ S
       
        return T
   

    def transform_pointcloud(self, pointcloud_batch):
            # Ensure point cloud is float
        pointcloud_batch = pointcloud_batch.float()

        # Apply scaling and rotation transformations
        transformation_matrix = self.get_transformation_matrix()
        transformed_pointcloud = torch.einsum('ij,bj->bi', transformation_matrix, pointcloud_batch.view(-1, 3))
       
        # Apply translation
        transformed_pointcloud += self.translation
       
        return transformed_pointcloud.view_as(pointcloud_batch)
   

   

    def GetPinspeckModel(self, ids, pinhole=False, return_vis=False):
           
        boxes = []
       
        for id in ids:
           
            x, y, z = self.points[id].cpu().numpy()
            box = x, y, z, x+self.grid_x_size, y+self.grid_y_size, z+self.grid_z_size
           
            boxes.append(box)
           
        scene = self.GetOccluderMesh(boxes)
        return self.GetForwardModelFromMesh(mesh=scene, pinhole=pinhole, return_location=False, return_vis=return_vis)
   

   
   
    def Get_location_model(self, ids=None, pinhole=True):
       
       
        if ids is None:
            ids = range(len(self.points))
           
           
        locations = []; values =[]
       
        for i in ids:
           
            x, y, z = self.points[i].cpu().numpy()
            box = x, y, z, x+self.grid_x_size, y+self.grid_y_size, z+self.grid_z_size
           
            mesh = self.GetOccluderMesh([box,])
           
            location, value = self.GetForwardModelFromMesh(mesh=mesh, pinhole=pinhole, return_location=True, return_vis=True)
           
           
            locations.append(location.astype(np.int16))
            # values.append(value)
            # print(locations)
           
        return locations#, values
       
       
   


    def GetSparseVisibility(self, ids=None, return_vis=True, pinhole=True, cast_to=torch.float16, batch_size=16):
       
        sparse_Model = []
        for i in range(0, len(self.points), batch_size):

            batch_Model = []

            rag = min(batch_size, len(self.points)-i)
         

            for j in range(i, i+rag):
                # print(j)

                x, y, z = self.points[j].cpu().numpy()
                box = x, y, z, x+self.grid_x_size, y+self.grid_y_size, z+self.grid_z_size

                mesh = self.GetOccluderMesh([box,])
                Model = self.GetForwardModelFromMesh(mesh=mesh, pinhole=pinhole, return_location=False, return_vis=return_vis).type(cast_to)

                batch_Model.append(Model.to_sparse().cpu())

            sparse_Model.append(torch.stack(batch_Model, 1))

        return sparse_Model
    

    def GetABarModel(self, ids=None, return_vis=False, pinhole=False, return_pin=1):

        
        
        A_bar = torch.zeros(self.Model.shape[0], self.Model.shape[1]*len(ids))
        sum_A_bar = 0
       
        
        for j, i in enumerate(ids):
            
            n = self.hidden_resolution**2
        
            x, y, z = self.points[i].cpu().numpy()
            box = x, y, z, x+self.grid_x_size, y+self.grid_y_size, z+self.grid_z_size
            
            scene = self.GetOccluderMesh([box])
        
            Model = self.GetForwardModelFromMesh(scene, pinhole=pinhole, return_vis=return_vis)
          
            sum_A_bar = sum_A_bar + Model
            
            A_bar[:, n*j:n*j+n] = Model
            
            
        if return_pin:
            return self.cast(A_bar), self.cast(sum_A_bar)
            
        if return_vis:
            A_bar = torch.cat([torch.ones_like(self.Model), -self.cast(A_bar)], -1)
        
        else:
            A_bar = torch.cat([self.Model, -self.cast(A_bar)], -1)
            
    
        return A_bar, self.cast(sum_A_bar)

   
       
       

    def OccMinMax(self, depth_difference = 3, plot=False, plot_range=5):
        """
        Function for Calculating the Depth of Occuler Space Using Line properties
        Args: To plot Oclcuder SPace using Matplot Lib
       
           
        Returns:
            y_min: The minimum value of the depth for occluders
            y_max: The minimum value of the depth for occluders
           
        """
        D = self.visible_y_range[0]
        x_line1, x_line2 = find_intercepting_line(self.hidden_x_range, self.visible_x_range, D)
        z_line1, z_line2 = find_intercepting_line(self.hidden_z_range, self.visible_z_range, D)

        x_int, yx_int = line_intersection_point(x_line1, x_line2)
        z_int, yz_int = line_intersection_point(z_line1, z_line2)


        xp_line1, xp_line2 = find_parallel_line(self.hidden_x_range, self.visible_x_range, D)
        zp_line1, zp_line2 = find_parallel_line(self.hidden_z_range, self.visible_z_range, D)

       
        y_avg = min(yx_int, yz_int)



        y_min, y_max = y_avg, y_avg + ((D-y_avg)/depth_difference)
        x_min, x_max = occ_range(D, (y_min, y_max), self.hidden_x_range, self.visible_x_range)
        z_min, z_max = occ_range(D, (y_min, y_max), self.hidden_z_range, self.visible_z_range)
       
       
        if plot:

            x = np.linspace(x_min, x_max, plot_range, dtype=self.default_dtype)
            y = np.linspace(y_min, y_max, plot_range, dtype=self.default_dtype)
            z = np.linspace(z_min, z_max, plot_range, dtype=self.default_dtype)

            print((x_min, x_max), (y_min, y_max), (z_min, z_max))

            x, y, z = np.meshgrid(x, y, z)

            xyz = np.column_stack([x.flatten("F"), y.flatten("F"), z.flatten("F")])


            fig = plt.figure(figsize=(12, 12))  
            ax = fig.add_subplot(111, projection='3d')

            ax.plot( np.array(x_line1)[:, 0], np.array(x_line1)[:, 1], np.zeros_like(x_line1)[:, 1]);
            ax.plot(np.array(x_line2)[:, 0], np.array(x_line2)[:, 1], np.zeros_like(x_line1)[:, 1])
            ax.plot(np.array(xp_line1)[:, 0], np.array(xp_line1)[:, 1], np.zeros_like(x_line1)[:, 1]);
            ax.plot(np.array(xp_line2)[:, 0], np.array(xp_line2)[:, 1], np.zeros_like(x_line1)[:, 1])
            ax.plot(np.zeros_like(x_line1)[:, 1], np.array(z_line1)[:, 1], np.array(z_line1)[:, 0]);
            ax.plot(np.zeros_like(x_line1)[:, 1], np.array(z_line2)[:, 1], np.array(z_line2)[:, 0])
            ax.plot(np.zeros_like(x_line1)[:, 1], np.array(zp_line1)[:, 1], np.array(zp_line1)[:, 0]);
            ax.plot(np.zeros_like(x_line1)[:, 1], np.array(zp_line2)[:, 1], np.array(zp_line2)[:, 0])
            ax.scatter(x_int, yx_int, 0)
            ax.scatter(0, yz_int , z_int, )
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])

       
            plt.axis("on")
            plt.show()
   
        return (y_min, y_max),  (xp_line1, xp_line2), (zp_line1, zp_line2)
   
   
   
def extract_bbox_info(bbox):
    if len(bbox) != 6:
        raise ValueError("Bounding box must have 6 elements (xmin, ymin, zmin, xmax, ymax, zmax)")

    xmin, ymin, zmin, xmax, ymax, zmax = bbox
   
    # Calculate the center
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    center_z = (zmin + zmax) / 2.0
   
    # Calculate the lengths
    length_x = xmax - xmin
    length_y = ymax - ymin
    length_z = zmax - zmin
   
    # Create tensors with gradients enabled
    center = torch.tensor([center_x, center_y, center_z], requires_grad=True)
    lengths = torch.tensor([length_x, length_y, length_z], requires_grad=True)
   
    return center, lengths
   
if __name__ == "__main__":
   
    import pickle
    import time
    device="cpu"
   
   
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
   
   
    # scene = PinsPeck(hidden_x_range=[0, .708], hidden_y_range=(0.0, 0.0), hidden_z_range=[0.03, 0.436],
    #             visible_x_range=[0.808, 1.5], visible_y_range=(1.076, 1.0), visible_z_range=[0.05, 0.05+0.692],
    #             hidden_resolution=10, visible_resolution=32, sub_resolution=1, device=device, grid_size=grid_size)
   

    grid_size =  [32, 16, 32] #Define Occluder Discritization
    M=(128, 128)        
    scene = PinsPeck(hidden_x_range=[0, .408], hidden_y_range=(0.0, 0.0), hidden_z_range=[0.1, 0.1+0.3085],
                visible_x_range=[0.218, 0.218+0.4672], visible_y_range=(1.076, 1.0), visible_z_range=[0.05, 0.05+0.4672],
                hidden_resolution=10, visible_resolution=M[0], sub_resolution=1, device=device, grid_size=grid_size)
   
   
    def point_as_occluder(point0, bounding_box_min, bounding_box_max):
       
       
            # Compute the bounding box of the mesh
            point0 = torch.from_numpy(point0)
            mesh_min = point0.min(0)[0] #torch.min(point0, axis=0)
            mesh_max = point0.max(0)[0] #(point0, axis=0)
       
            mesh_dims = mesh_max - mesh_min
            bounding_box_dims = bounding_box_max - bounding_box_min
       
            longest_dim_index = torch.argmax(mesh_dims)
       
            ratio = mesh_dims/mesh_dims[longest_dim_index]
            ratio[ratio<0.5] = 0.5
            rat = bounding_box_dims[longest_dim_index]*ratio

            point0 = ((point0 - mesh_min)/(mesh_max-mesh_min))*(rat) + bounding_box_min

            return point0
   
   
   

    def generate_cuboid(center, size, n_points):
        x = np.random.uniform(center[0] - size[0]/2, center[0] + size[0]/2, n_points)
        y = np.random.uniform(center[1] - size[1]/2, center[1] + size[1]/2, n_points)
        z = np.random.uniform(center[2] - size[2]/2, center[2] + size[2]/2, n_points)
        return x, y, z

    # Define the number of points for each part of the chair
    n_points_leg = 250
    n_points_seat = 1000
    n_points_back = 500

    # Generate point clouds for the chair parts
    # 4 legs
    leg1_x, leg1_y, leg1_z = generate_cuboid(center=[0.1, 0.1, 0.25], size=[0.1, 0.1, 0.5], n_points=n_points_leg)
    leg2_x, leg2_y, leg2_z = generate_cuboid(center=[0.9, 0.1, 0.25], size=[0.1, 0.1, 0.5], n_points=n_points_leg)
    leg3_x, leg3_y, leg3_z = generate_cuboid(center=[0.1, 0.9, 0.25], size=[0.1, 0.1, 0.5], n_points=n_points_leg)
    leg4_x, leg4_y, leg4_z = generate_cuboid(center=[0.9, 0.9, 0.25], size=[0.1, 0.1, 0.5], n_points=n_points_leg)

    # Seat
    seat_x, seat_y, seat_z = generate_cuboid(center=[0.5, 0.5, 0.55], size=[1.0, 1.0, 0.1], n_points=n_points_seat)

    # Back
    back_x, back_y, back_z = generate_cuboid(center=[0.5, 0.9, 1.0], size=[1.0, 0.1, 0.9], n_points=n_points_back)

    # Combine the parts into a single point cloud
    x = np.concatenate([leg1_x, leg2_x, leg3_x, leg4_x, seat_x, back_x])
    y = np.concatenate([leg1_y, leg2_y, leg3_y, leg4_y, seat_y, back_y])
    z = np.concatenate([leg1_z, leg2_z, leg3_z, leg4_z, seat_z, back_z])
   
    xyz = np.stack([x, y, z], -1)
   
    boxes = np.array(scene.generate_sweeping_bounding_boxes(0.1, 0.05, 0.1))


    print(boxes)
    box = boxes[np.random.randint(0, len(boxes))]
    points = point_as_occluder(xyz, box[:3], box[3:] )
   
   
     # Creating a new figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    # scatter = ax.scatter(x, y, z)
    scatter = ax.scatter(scene.hidden[:, 0, 0].cpu(), scene.hidden[:, 0, 1].cpu(), scene.hidden[:, 0, 2].cpu())
    scatter = ax.scatter(scene.visible[:, 0].cpu(), scene.visible[:, 1].cpu(), scene.visible[:, 2].cpu())
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    # Labeling axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()
   
   
    mode = scene.GetForwardModelFromPoints(points )


    initial_image = torch.ones(10, 10)
   
    def plot_side_by_side_with_labels():
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
       
        # Initialize the left plot with the first column as placeholder
        column_data = np.reshape(mode[:, 0].numpy(), M,  "F")
        img_display = ax1.imshow(column_data, cmap='viridis', origin="lower")
        ax1.axis('off')
       
        # Display the 10x10 grid on the right with labels
        ax2.imshow(initial_image, cmap='gray', origin="lower")
        ax2.set_xticks(np.arange(-.5, 10, 1))
        ax2.set_yticks(np.arange(-.5, 10, 1))
        ax2.grid(which='both')
       
        # Annotate each cell with its index
        for i in range(10):
            for j in range(10):
                ax2.text(i, j, f'({i},{j})', ha='center', va='center', color='white')
       
        ax2.axis('off')
       
        def on_click(event):
            # Determine the x and y indices of the clicked pixel on the right plot
            if event.inaxes == ax2:
                x, y = int(event.xdata), int(event.ydata)
               
                # Convert the (x, y) to a flat index
                index = x * 10 + y
               
                # Retrieve the corresponding column from the matrix
                column_data = np.reshape(mode[:, index].numpy(), M,  "F")
               
                # Update the left plot
                img_display.set_data(column_data,)
                fig.canvas.draw()
           
        # Connect the click event to the on_click function
        fig.canvas.mpl_connect('button_press_event', on_click)
       

        plt.tight_layout()
        plt.show()

    plot_side_by_side_with_labels()