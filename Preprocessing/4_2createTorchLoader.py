
import os
import pickle
import networkx as nx
import numpy as np
import torch
import torch_geometric

curfolder = os.getcwd()

textures_info = {
    # 30: 1,
    # 45: 1,
    # 90: 1,
    # 160: 1,
    # 200: 1,
    250: 1
}

textures = [250]# ,45,90,160,200] # textures to be combined
avg_mode = True # True if it uses averaged FIP over grain, False if it uses maximum sub-band FIP
fraction = 1
seed = 42


datalist = [] # elements: {'x': [.., .., .., ...], 'edge_index':[[...],[...]], 'e': [.., .., .., ...], 'fip': [..., ..., ...]}

for texture in textures:
    numSVEs = textures_info[texture]
    ## Fill in datalist while doing scaling

    for i in range(numSVEs):
        graph_file = f'{curfolder}\\{texture}\\sve_{i}'
        fip_file = f'{curfolder}\\{texture}\\fip_avg_{i}.csv' if avg_mode else f'{curfolder}\\fip_{i}.csv'

        G = nx.read_gpickle(graph_file)
        data = torch_geometric.utils.from_networkx(G)
        x = data.x
        edge_index = data.edge_index

        fip = np.loadtxt(fip_file, delimiter=',')[:,1]

        with open(file=f'{curfolder}\\feat_scaler.pickle', mode='rb') as f:
            nfeat_scaler = pickle.load(f)
        x = nfeat_scaler.transform(x)

        if avg_mode:
            with open(file=f'{curfolder}\\fip_avg_scaler.pickle', mode='rb') as f:
                fip_scaler = pickle.load(f)
            fip = fip_scaler.transform(fip[:,None])
        else:
            with open(file=f'{curfolder}\\fip_scaler.pickle', mode='rb') as f:
                fip_scaler = pickle.load(f)
            fip = fip_scaler.transform(fip[:,None])
            

        # rewrite data with type casting
        data.x = torch.from_numpy(x).float()
        data.edge_index = edge_index
        data.fip = torch.from_numpy(fip).float()

        datalist.append(data)

        print(f'{texture},{i}')

if fraction < 1:
    pass
    # ## Shuffle, split and save
    # np.random.seed(seed)
    # bnd_idx = int(len(datalist)*fraction)

    # # shuffle
    # rand_ids = np.random.choice(len(datalist), len(datalist), replace=False)
    # datalist_new = [datalist[i] for i in rand_ids]

    # # split
    # train_datalist = datalist_new[:bnd_idx]
    # val_datalist = datalist_new[bnd_idx:]

    # fname = f"{'_'.join(map(str, textures))}_quat_frac{round(fraction*100)}.pickle"
    # with open(f'{curfolder}\\loaders\\{fname}', 'wb') as f:
    #     pickle.dump(train_datalist, f, protocol=pickle.HIGHEST_PROTOCOL)

    # fname = f"{'_'.join(map(str, textures))}_quat_frac{round((1-fraction)*100)}.pickle"
    # with open(f'{curfolder}\\loaders\\{fname}', 'wb') as f:
    #     pickle.dump(val_datalist, f, protocol=pickle.HIGHEST_PROTOCOL)

else:

    fname = "250single.pickle"
    with open(f'{curfolder}\\loaders\\{fname}', 'wb') as f:
        pickle.dump(datalist, f, protocol=pickle.HIGHEST_PROTOCOL)