import torch
from collections import defaultdict
import math

from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
def inception_score(images, batch_size=100, device=torch.device('cpu')):
    """
    :param images: generated images, tensor of shape [num_images, *]
    :param device: torch.Device object
    :return: inception score, float
    """
    metric = InceptionScore(device=device, normalize=True)

    num_batches = math.ceil(images.size(0) / batch_size)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        stop_idx = min((batch_idx + 1) * batch_size, images.size(0))
        image_batch = images[start_idx:stop_idx]
        metric.update(image_batch)

    return metric.compute()

def fid_score(images_pred, images_true, batch_size=100, device=torch.device('cpu')):
    """
    :param images_pred: generated images, tensor of shape [num_images, *]
    :param images_true: ground truth images, tensor of shape [num_images, *]
    :param device: torch.Device object
    :return: frechet inception distance, float
    """
    metric = FrechetInceptionDistance(device=device, normalize=True)

    num_batches = math.ceil(images_pred.size(0) / batch_size)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        stop_idx = min((batch_idx + 1) * batch_size, images_pred.size(0) - 1)
        image_pred_batch = images_pred[start_idx:stop_idx]
        image_true_batch = images_true[start_idx:stop_idx]
        metric.update(image_pred_batch, real=False)
        metric.update(image_true_batch, real=True)

    return metric.compute()

class MetricAccumulator:
    """
    Object for accumulating values of metric during an epoch
    """
    def __init__(self):
        self.concat_keys = []
        self.reset()

    def update(self, metrics):
        """
        add new values for metrics to be updated
        :param metrics: new metric values, dict of floats
        """
        for key, value in metrics.items():
            # if key is to be excluded from averaging, save in list
            if key in self.concat_keys:
                self.metrics[key] = [] if self.metrics[key] == 0 else self.metrics[key]
                self.metrics[key] += [value]
            else:
                self.metrics[key] += value
        self.counter += 1

    def get_average(self):
        """
        compute the average of all metrics accumulated so far
        :return: average values for all metrics, dict of floats
        """
        for key in self.metrics.keys():
            if not key in self.concat_keys:
                self.metrics[key] /= self.counter
        return self.metrics

    def reset(self):
        """
        call at the end of an epoch to reset all entries
        """
        self.metrics = defaultdict(lambda: 0)
        self.counter = 0

    def exclude_keys_from_average(self, keys):
        """
        exclude some keys from averaging, collect them in a list instead
        :param keys: keys to be excluded, list of str
        """
        self.concat_keys += keys

def mutual_information_data_rep(cov_matrix_marginal_distr: torch.Tensor,
                                cov_matrix_cond_distr: torch.Tensor,
                                lin_map: torch.Tensor):
    """
    Computes mutual information between:
    p(x) = N(m_x, cov_matrix_marginal_distr), and
    q(z|x) = N(lin_map . x + b, cov_matrix_cond_distr).
    The mutual information reads:
    MI = 0.5 * log(det(cov_matrix_marginal_distr)) +
         0.5 * log(det(lin_map * cov_matrix_marginal_distr * lin_map^T)) -
         0.5 * log(det(cov_full))
    where cov_full is the covariance matrix of the (Gaussian) random variable y = (x, z)
    """

    m1 = torch.matmul(cov_matrix_marginal_distr, torch.t(lin_map))
    m2 = torch.matmul(lin_map, cov_matrix_marginal_distr)
    m3 = cov_matrix_cond_distr + torch.matmul(lin_map, m1)
    m1 = torch.cat([cov_matrix_marginal_distr, m1], dim=-1)
    m2 = torch.cat([m2, m3], dim=-1)
    cov_full = torch.cat([m1, m2], dim=0)

    epsilon = torch.tensor(1e-10, device=m1.device)
    zeros = torch.tensor(0.0, device=m1.device)
    det_m3 = torch.det(m3)
    log_det_m3 = torch.log(torch.where(det_m3 > zeros, det_m3, epsilon))
    det_cov_mat_marginal_distr = torch.det(cov_matrix_marginal_distr)
    log_det_cov_mat_marginal_distr = torch.log(torch.where(det_cov_mat_marginal_distr > zeros,
                                                           det_cov_mat_marginal_distr,
                                                           epsilon))
    det_cov_full = torch.det(cov_full)
    log_det_cov_full = torch.log(torch.where(det_cov_full > zeros, det_cov_full, epsilon))
    mi = 0.5 * log_det_m3 + 0.5 * log_det_cov_mat_marginal_distr - 0.5 * log_det_cov_full
    # mi = 0.5 * torch.log(log_det_m3) + 0.5 * torch.log(det_cov_mat_marginal_distr) - 0.5 * torch.log(log_det_cov_full)
    # mi = 0.5 * torch.log(det_m3 * det_cov_mat_marginal_distr / det_cov_full)
    return mi