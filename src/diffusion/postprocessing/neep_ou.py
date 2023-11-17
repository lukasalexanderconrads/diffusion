from diffusion.postprocessing.neep import *

"""
experiment: model selection with ornstein-uhlenbeck process
i have a directory of models. i wanna compute:
- real epr
- real ep
- estimate epr via simple on test set
- estimate epr via var on test set
- estimate ep via simple on test set
- average relative error of simple epr
- average relative error of var epr
i wanna plot:
model sizes on x axis
ep values on y axis
specifically:
plot 1:
- real ep
- estimate ep via simple on test set
- estimate ep via var on test set
plot 2:
- avg relative error of simple epr
- avg relative error of var epr
"""


def plot_dat_shit(result_dir, model_name_list, label_list):

    trainer_list = get_trainer_list(result_dir, model_name_list)

    exact_ep_list = []
    simple_ep_list = []
    var_ep_list = []
    for trainer in trainer_list:
        data_loader = trainer.data_loader

        ### EXACT EPR
        exact_epr = get_exact_epr(data_loader)
        exact_ep =

        ### CURRENT
        get_j_t(trainer)




def get_exact_epr(data_loader):
    exact_epr = data_loader.train.dataset.exact_epr
    return exact_epr

model_name_list = ['mv_ou_process/_dim_3_T_10_num_steps_10000_num_samples_1000_max_cond_number_5/_model_layer_dims_(16, 16)/0802-100639']
label_list = ['']

trainer_list = get_trainer_list(result_dir, model_name_list)
plot_epr(trainer_list, label_list)
plt.xlim(-.1, 10)
plt.title('$dt = .001$')
plt.show()
