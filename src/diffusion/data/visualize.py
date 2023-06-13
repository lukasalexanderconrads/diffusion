from datasets import TrajectoryDatasetAE
import matplotlib.pyplot as plt
import torch



if __name__ == '__main__':

    #dataset = TrajectoryDatasetAE(path='/raid/data/neep/ppca/_loader_batch_size_32_optimizer_lr_0.005/0602-095414')
    dataset = TrajectoryDatasetAE(path='/raid/data/neep/neuron/1d/_trainer_lr_scheduler_increase_per_epoch_0.008/')
    #dataset = TrajectoryDatasetAE(path='/raid/data/neep/ppca/_loader_batch_size_32_optimizer_lr_0.0005/0602-135144')

    data = dataset[:]['data']
    latent_traject = data[1, :, 0]
    time = dataset[0]['time_point']

    for trajectory in data:
        plt.plot(time, trajectory[:, 0, 0])
    # plt.plot(time, latent_traject[:, 2])
    plt.show()


    latent_start = data[:, 0, 0]
    latent_end = data[:, -1, 0]
    # print(torch.mean(latent_start, dim=0))
    # print(torch.cov(latent_start.T))
    print('final latent distribution')
    print('mean:')
    print(torch.mean(latent_end, dim=0))
    print('covariance:')
    print(torch.cov(latent_end.T))

    #print('mi difference:', dataset.mi[-1] - dataset.mi[0])
