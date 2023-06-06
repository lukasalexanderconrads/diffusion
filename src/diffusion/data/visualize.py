from datasets import TrajectoryDatasetAE
import matplotlib.pyplot as plt
import torch



if __name__ == '__main__':

    dataset = TrajectoryDatasetAE(path='/raid/data/neep/ae/_loader_batch_size_64_optimizer_lr_0.005/0605-122439')

    latent_traject = dataset[5]['data'][:, 0]
    print(latent_traject[:10], latent_traject[10:])
    time = dataset[0]['time_point']

    plt.plot(time, latent_traject[:, 0])
    plt.plot(time, latent_traject[:, 1])
    plt.plot(time, latent_traject[:, 2])
    plt.show()

    latent_start = dataset[:]['data'][0, 0]
    latent_end = dataset[:]['data'][-1, 0]
    # print(torch.mean(latent_start, dim=0))
    # print(torch.cov(latent_start.T))
    print('final latent distribution')
    print('mean:')
    print(torch.mean(latent_end, dim=0))
    print('covariance:')
    print(torch.cov(latent_end.T))

    #print('mi difference:', dataset.mi[-1] - dataset.mi[0])
