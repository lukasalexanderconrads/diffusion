import os
import io
import sys
import json
import click
import logging
import numpy as np
import torch

from diffusion.models.data_models import LatentLinearGaussian
from diffusion.utils.helpers import read_yaml
from diffusion.utils.metrics import *

@click.command()
@click.option('-c', '--config', 'cfg_path', required=True,
              type=click.Path(exists=True), help='path to data config file')

def main(cfg_path):

    params = read_yaml(cfg_path)
    out_dir = params['save_dir']
    use_hidden_variable_model = params['use_hidden_variable_model']
    use_random_lin_map = params['use_random_lin_map']
    train_size = params['train_size']
    test_size = params['test_size']
    validate_size = params['validation_size']
    seed = params['seed']

    pca_dimension = params['pca_dimension']

    # Gaussian parameters
    mean_z = params['mean_z']
    std_dev_z = params['std_dev_z']
    std_dev_x = params['std_dev_x']

    # Parameters for pre-defined latent-to-data map
    theta = params['rotation_angle_y']
    phi = params['rotation_angle_x']
    sigma = params['shear_mapping_1']
    rho = params['shear_mapping_2']

    data_dim = params['data_dim']

    os.makedirs(out_dir, exist_ok=True)
    lin_gaussian = LatentLinearGaussian(theta, phi, sigma, rho,
                                        mean_z, std_dev_z, std_dev_x,
                                        use_random_lin_map, use_hidden_variable_model,
                                        data_dim)

    np.random.seed(seed)
    generator = lin_gaussian.sample(train_size, int(1000))

    train_path = os.path.join(out_dir, 'train.csv')
    train_latent_path = os.path.join(out_dir, 'train_latent.csv')
    remove_file_if_exists(train_path)
    remove_file_if_exists(train_latent_path)
    with io.open(train_path, 'a+') as f:
        with io.open(train_latent_path, 'a+')as f_z:
            for x, z in generator:
                np.savetxt(f, x, delimiter=',', fmt='%1.6f')
                np.savetxt(f_z, z, delimiter=',', fmt='%1.6f')

    if test_size is not None:
        generator = lin_gaussian.sample(test_size, int(1000))

        test_path = os.path.join(out_dir, 'test.csv')
        test_latent_path = os.path.join(out_dir, 'test_latent.csv')
        remove_file_if_exists(test_path)
        remove_file_if_exists(test_latent_path)
        with io.open(test_path, 'a+') as f:
            with io.open(test_latent_path, 'a+') as f_z:
                for x, z in generator:
                    np.savetxt(f, x, delimiter=',', fmt='%1.6f')
                    np.savetxt(f_z, z, delimiter=',', fmt='%1.6f')

        generator = lin_gaussian.sample(validate_size, int(1000))

        validate_path = os.path.join(out_dir, 'valid.csv')
        remove_file_if_exists(validate_path)
        validate_latent_path = os.path.join(out_dir, 'valid_latent.csv')
        remove_file_if_exists(validate_latent_path)
        with io.open(validate_path, 'a+') as f:
            with io.open(validate_latent_path, 'a+') as f_z:
                for x, z in generator:
                    np.savetxt(f, x, delimiter=',', fmt='%1.6f')
                    np.savetxt(f_z, z, delimiter=',', fmt='%1.6f')


    cov_matrix_posterior, proj_posterior = compute_max_likelihood_pca(out_dir, pca_dimension,
                                                                      lin_gaussian.data_cov_matrix,
                                                                      lin_gaussian.mean_x)

    # mutual information calculation:
    mi_ml = mutual_information_data_rep(torch.from_numpy(lin_gaussian.data_cov_matrix),
                                        torch.from_numpy(cov_matrix_posterior),
                                        torch.from_numpy(proj_posterior))

    if lin_gaussian.use_hidden_variable_model:
        cov_z = np.zeros((lin_gaussian.latent_dim, lin_gaussian.latent_dim))
        np.fill_diagonal(cov_z, lin_gaussian.std_dev_z[0] * lin_gaussian.std_dev_z[0])
        cov_x = np.zeros((lin_gaussian.data_dim, lin_gaussian.data_dim))
        np.fill_diagonal(cov_x, lin_gaussian.std_dev_x[0] * lin_gaussian.std_dev_x[0])
        mi_gt = mutual_information_data_rep(torch.from_numpy(cov_z),
                                            torch.from_numpy(cov_x),
                                            torch.from_numpy(lin_gaussian.lin_map))
    else:
        mi_gt = None
    print('max-likelihood MI(X, Z) =', mi_ml)
    print('ground-truth MI(X, Z) =', mi_gt)


def compute_max_likelihood_pca(path, pca_dimension, ground_truth_data_cov_matrix, ground_truth_data_mean):
    data_ = np.genfromtxt(os.path.join(path, 'train.csv'), delimiter=',')
    data_dim = data_.shape[-1]
    data_cov_matrix = np.cov(data_.T)
    ev, eig = np.linalg.eig(data_cov_matrix)
    # sort eigenvalues and take the fist 'pca_dimension'
    idx = ev.argsort()[::-1]
    ev = ev[idx]
    eig = eig[:, idx][:, :pca_dimension]
    # compute the max-likelihood mean
    mean_ml = np.mean(data_, axis=0)  # [data_dim]
    # compute the max-likelihood variance
    var_ml = np.sum(ev[pca_dimension:]) / (data_dim - pca_dimension)
    # compute the max-likelihood projection:
    L = np.zeros((pca_dimension, pca_dimension))
    np.fill_diagonal(L, ev[:pca_dimension] - var_ml)
    lin_proj = np.matmul(eig, np.sqrt(L))  # [data_dim, pca_dim]
    # compute the max-likelihood covariance matrix
    cov_ml = np.matmul(lin_proj, lin_proj.T) + var_ml * np.eye(data_dim)

    # exact posterior/encoder
    m1 = np.matmul(lin_proj.T, lin_proj) + var_ml * np.eye(pca_dimension)
    m1 = np.linalg.inv(m1)
    enc_projection = np.matmul(m1, lin_proj.T)  # encoder projection
    enc_cov = var_ml * m1

    # store into file
    max_likelihood = {'mean_ml': mean_ml.tolist(),
                      'covariance_matrix_ml': cov_ml.tolist(),
                      'encoder_proj_ml': enc_projection.tolist(),
                      'encoder_covariance_ml': enc_cov.tolist(),
                      'ground_truth_data_cov_matrix': ground_truth_data_cov_matrix.tolist(),
                      'ground_truth_data_mean': ground_truth_data_mean.tolist()}
    path_ = os.path.join(path, 'max_likelihood_sol.json')
    remove_file_if_exists(path_)
    with open(path_, 'w', encoding='utf-8') as f:
        json.dump(max_likelihood, f, ensure_ascii=False, indent=4)

    return enc_cov, enc_projection


def remove_file_if_exists(path):
    if os.path.exists(path):
        logging.info("Deleting old file" + path)
        os.remove(path)

if __name__ == '__main__':
    main()
