
from matplotlib import pyplot as plt
import torch.nn.functional as F
from utils_training import getLoader
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.cm as cm

def parity_plot(model, weights, loader_dir):

    model.load_state_dict(weights)

    model.eval()

    loader = getLoader(datalist_dir=loader_dir, batch_fraction=1)
    iterer = iter(loader)
    batch = next(iterer)

    y = model(batch)

    XX = batch.fip.detach().numpy()
    YY = y.detach().numpy()

    plt.figure(figsize=(10,10))
    plt.scatter(XX, YY, s=10, color=cm.winter(1), alpha=0.4, marker='x')
    # plt.colorbar()
    plt.plot([-3,2],[-3,2],color='black')
    plt.xlim((-4,5))
    plt.ylim((-4,5))
    print(F.mse_loss(y, batch.fip))
    print(np.mean(np.abs(y.detach().numpy() - batch.fip.detach().numpy()/batch.fip.detach().numpy())))
    print(F.l1_loss(y, batch.fip))
    print(r2_score(y.detach().numpy(), batch.fip.detach().numpy()))

def multi_parity_plot(models, weightss, loader_dirs):
    assert(len(models) == len(weightss) and len(weightss) == len(loader_dirs))
    for i in range(len(models)):
        model = models[i]
        weights = weightss[i]
        loader_dir = loader_dirs[i]
        parity_plot(model,weights,loader_dir)