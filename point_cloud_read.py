import numpy as np
import open3d

path = 'airplane_aeroplane_plane_2691156_1a04e3eab45ca15dd86060f189eb1331_10000_2pc.ply'
cloud_path_header = str('/mnt/c/Data/PointNetWork/AllClouds10k/')
num_points = 2000
sample_ratio = 10000//num_points
try:
    # path = path.numpy()
    # path = np.array2string(path)
    # for character in "[]]'":
    #     path = path.replace(character, '')
    #     # print(path)
    # path = path[1:]
    path = cloud_path_header + path
    print(path) 
    cloud = open3d.io.read_point_cloud(path)
    cloud = cloud.uniform_down_sample(every_k_points=sample_ratio)
except:
    cloud = np.random.rand((num_points,3))
    path = 'ERROR IN PCREAD: Transformation from Tensor to String Failed'
finally:
    cloud = cloud.points
    cloud = np.asarray([cloud])[0]
print(len(cloud[0]))