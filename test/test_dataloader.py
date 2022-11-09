import os

from diffusion.data.dataloaders import *
import torch
import matplotlib.pyplot as plt

path = '/cephfs/projects/ml2r_entropy/pca/trajectories_train/training_data_loader_batch_size_32_optimizer_lr_0.005/0208_113149'
#path = '/cephfs/projects/ml2r_entropy/pca/trajectories_test/training_sgd_data_loader_batch_size_32_optimizer_lr_0.05/0208_120221'

mi = torch.load(os.path.join(path, 'mi.pt'))
plt.plot(range(len(mi)), mi)
plt.show()
print(mi)
exit()

loader = DataLoaderTrajectoryAE(path=path)

for minibatch in loader.train:
    time_points = minibatch['time_point']
    plt.plot(torch.arange(time_points.size(1)), time_points[0, :, 0])
    plt.show()
    break

