import numpy as np
import pcl
p = pcl._pcl.PointCloud_PointXYZI(10)  # "empty" point cloud
a = np.asarray(p)       # NumPy view on the cloud
a[:] = 0                # fill with zeros
print(p[3])             # prints (0.0, 0.0, 0.0)
a[:, 0] = 1             # set x coordinates to 1
print(p[3])             # prints (1.0, 0.0, 0.0)import pcl

for i in range(0, 10):
    a[i, :2] = 0.1 * i

viewer = pcl.CloudViewing()

while not viewer.WasStopped():
    for i in range(0, 10):
        a[i, 0] = a[i, 0] + 0.01
    
    print(p[3])

    viewer.ShowGrayCloud(p)