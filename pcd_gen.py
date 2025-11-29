import open3d as o3d
import numpy as np

planes_model = np.random.rand(100, 3) * 2 - 1
planes_model /= np.linalg.norm(planes_model, axis=1)
ds = np.random.rand(planes_model.shape[0], 1) * 20 - 10
planes_model = np.append(planes_model, ds)

