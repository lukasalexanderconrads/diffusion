from datasets import TrajectoryDatasetAE
import matplotlib.pyplot as plt



if __name__ == '__main__':

    dataset = TrajectoryDatasetAE(path='/raid/data/neep/ppca/_loader_batch_size_32_optimizer_lr_0.0005/0602-135144')

    latent_traject = dataset[0]['data'][:, 0]
    time = dataset[0]['time_point']

    plt.plot(time, latent_traject[:, 0])
    plt.plot(time, latent_traject[:, 1])
    plt.plot(time, latent_traject[:, 2])
    plt.show()