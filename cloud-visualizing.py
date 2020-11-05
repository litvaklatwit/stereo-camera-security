import pcl
import numpy as np
import time

#p = pcl._pcl.PointCloud_PointXYZI(10)  # "empty" point cloud
#p = pcl.PointCloud(10)  # "empty" point cloud
#a = np.asarray(p)       # NumPy view on the cloud
#a[:] = 0                # fill with zeros
a = np.zeros((10, 3), dtype=np.float32)

for i in range(0, 10):
    a[i, :2] = 0.1 * i

vis = pcl.pcl_visualization.PCLVisualizering()
vis.AddCoordinateSystem(1.0)
vis.InitCameraParameters()

while not vis.WasStopped():
    for i in range(0, 10):
        a[i, 0] = a[i, 0] + 0.01

    p = pcl.PointCloud(a)

    vis.RemoveAllPointClouds(0)
    vis.AddPointCloud(p)

    vis.SpinOnce()
    time.sleep(0.016)