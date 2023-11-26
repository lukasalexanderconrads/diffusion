# diffusion

install the repository via pip with
```pip install -e .```

Generating data:

Generating Ornstein-Uhlenbeck trajectories:

```python scripts/generate_ou_trajectory.py -c generate_data/mv_ou_process.yaml```

Generating gaussian data with latent variable:

```python scripts/generate_linear_gaussian.py -c generate_data/gaussian_linear.yaml```

Generating LVM latent trajectories:

```python scripts/generate_train_trajectory.py -c generate_data/ppca.yaml```

Training a model:

```python scripts/train_model.py -c config.yaml```

config files for latent variable models are in ```experiments/autoencoder/```

config files for neep experiments are in ```experiments/entropy/```


