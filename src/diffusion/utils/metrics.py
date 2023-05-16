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
