import torch
from ignite.metrics import InceptionScore
def inception_score(images, batch_size=100, device=torch.device('cpu')):
    metric = InceptionScore(device=device)

    num_batches = images.size(0) // batch_size + 1
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        stop_idx = min((batch_idx + 1) * batch_size, images.size(0) - 1)
        image_batch = images[start_idx:stop_idx]
        metric.update(image_batch)

    return metric.compute()
