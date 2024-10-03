import numpy as np
import sys
import os
import math

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
    rotation_matrix = np.eye(3)
    
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)
    
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
    
    rotated_pointcloud = np.matmul(pointcloud, rotation_matrix)
    return rotated_pointcloud

def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result

def run(pcl, xml_path):

   

    xml_head = \
    """
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>
            
            <sampler type="ldsampler">
                <integer name="sampleCount" value="144"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="512"/>
                <integer name="height" value="512"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>
        
        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>
        
    """

    xml_ball_segment = \
    """
        <shape type="sphere">
            <float name="radius" value="0.025"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

    xml_tail = \
    """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
        </shape>
        
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="6,6,6"/>
            </emitter>
        </shape>
    </scene>
    """

    def colormap(x,y,z):
        vec = np.array([x,y,z])
        vec = np.clip(vec, 0.001,1.0)
        norm = np.sqrt(np.sum(vec**2))
        vec /= norm
        return [vec[0], vec[1], vec[2]]
    xml_segments = [xml_head]

    
    pcl = standardize_bbox(pcl, 2048)
    pcl = pcl[:,[2,0,1]]
    pcl[:,0] *= -1
    pcl[:,2] += 0.0125

    for i in range(pcl.shape[0]):
        color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
        xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    with open(xml_path, 'w') as f:
        f.write(xml_content)


# Get paths from command line arguments
npy_file_path = sys.argv[1]
xml_output_dir = sys.argv[2]

# Load the .npy file
pcll = np.load(npy_file_path, allow_pickle=True)[:, :, :,  [0,2,1]]
print(pcll.shape)
pcl = [pcll[p] for p in range(0,  len(pcll), 16)]


pcl = pcl + [rotate_pointcloud(pcll[-1], i, 1 ) for i in range(0, 360, 15)]
pcl = np.stack(pcl)[:, 0]
# print(pcl.shape)
# Process each item in the list and generate XML files
for i in range(6, len(pcl)):
    xml_path = os.path.join(xml_output_dir, f"{i}.xml")

    from occluder.datasets.pointcloud import PointCloud

    # import trimesh

    # x = trimesh.points.PointCloud(vertices=standardize_bbox(pcl[-1][0], 15000))
    # n = rotate_pointcloud(pcl[-1][0], 45, 1 )
    # n = trimesh.points.PointCloud(vertices=standardize_bbox(n, 15000))
    # # print(x), pcl[i][0]
    # n.vertices[:, 0] += 0.5
    # (x+n).show()



    pc = PointCloud(coords=pcl[i], channels={})
    pc = pc.farthest_point_sample(2048).coords
    print(pc.shape)
    run(pc, xml_path)