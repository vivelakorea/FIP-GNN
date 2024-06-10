import os
from utils_training import train
from utils_training import GNN
from utils_training import getLoader
import torch
import datetime

torch.set_num_threads(48+12+6)


t = 3
tr_val_set = [
            #  ('30_quat_frac90', '30_quat_frac10'),
  
                (f'30_top{t}_frac90', f'30_top{t}_frac10'),
            #   ('30_frac100', '45_frac100'),

            #   ('30_45_frac90', '30_45_frac90'),
            #   ('30_45_frac100', '90_frac100'),

            #   ('30_45_90_frac90', '30_45_90_frac10'),
            #   ('30_45_90_frac100', '160_frac100'),

            #   ('30_45_90_160_frac90', '30_45_90_160_frac10'),
            #   ('30_45_90_160_frac100', '200_frac100'),

            #   ('30_45_90_160_200_frac90', '30_45_90_160_200_frac10'),
            #   ('30_45_90_160_200_frac100', '250_frac100')
              ]

###################################################################################################################

print(f'started: {datetime.datetime.now()}')

# for k in [1,2,3,4,5]:
#     for n in [1,2,4,8,16,32]:
n = 16
k = 4

for training, validation in tr_val_set:
    # training = '30_frac90'
    # validation = '30_frac10'
    # n = 32
    # k = 4

    ###################################################################################################################

    curfolder = os.getcwd()
    trainlist = f'{curfolder}\\datalist\\{training}.pickle'
    vallist = f'{curfolder}\\datalist\\{validation}.pickle'
    learningcurvefile = f'{curfolder}\\learning_curves\\{training}_{validation}_n_{n}_k_{k}_avg.csv'
    weightsfile = f'{curfolder}\\trained_weights\\{training}weights_top{t}_n_{n}_k_{k}_avg.pt'

    train_loader = getLoader(datalist_dir=trainlist, batch_fraction=0.2)

    val_loader = getLoader(datalist_dir=vallist, batch_fraction=1.0)


    train_params = {
        'opt_name': 'Adam',
        'n_epoch': 1000,
        'lr': 5.e-2,
        'weight_decay': 1.e-4,
        'loss_fname': 'mseLoss',
        'lr_decay_rate': 0.8
    }


    model = GNN(n,k,top=t)

    train(model=model,
        train_params=train_params,
        train_loader=train_loader,
        val_loader=val_loader,
        logfile_dir=learningcurvefile,
        print_mode=False
        )

    torch.save(model.state_dict(), weightsfile)

    print(f'{training}, {validation}, k={k}, n={n} ended: {datetime.datetime.now()}')