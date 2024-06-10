import os
import numpy as np
import pickle
import networkx as nx
import torch_geometric
from sklearn.discriminant_analysis import StandardScaler

    
curfolder = os.getcwd()
textures = [30, 45, 90, 160, 200, 250]
numSVEs = [200, 200, 200, 100, 4, 2]

feature_data = np.zeros((0, 12))
fip_data = np.zeros((0, 1))
fip_avg_data = np.zeros((0,1))

for i in range(6):
    texture = textures[i]
    numSVE = numSVEs[i]
    for ith_sve in range(numSVE):

        # feature
        graph_file = f'{curfolder}\\{texture}\\sve_{ith_sve}'

        G = nx.read_gpickle(graph_file)
        data = torch_geometric.utils.from_networkx(G)
        feature_data = np.vstack((feature_data, data.x))

        # fip    
        fip_file = f'{curfolder}\\{texture}\\fip_{ith_sve}.csv'
        fips = np.loadtxt(fip_file, delimiter=',')
        fips = np.column_stack([fips[:,1]]) # transpose 1d array to column vector
        fip_data = np.vstack((fip_data, fips))

        # fip_avg
        fip_avg_file = f'{curfolder}\\{texture}\\fip_avg_{ith_sve}.csv'
        fip_avgs = np.loadtxt(fip_avg_file, delimiter=',')
        fip_avgs = np.column_stack([fip_avgs[:,1]]) # transpose 1d array to column vector
        fip_avg_data = np.vstack((fip_avg_data, fip_avgs))

        print(f'{texture},{ith_sve}')
    


# all_data = np.hstack((feature_data, fip_data, fip_avg_data))
# fout = open(f'{curfolder}\\all_XY_merged.csv', 'w')
# fout.write('\n'.join(','.join('%f' %x for x in y) for y in all_data)+'\n')
# fout.close()

feat_scaler = StandardScaler()
feat_scaler.fit(feature_data)

with open(file=f'{curfolder}\\feat_scaler.pickle', mode='wb') as f:
    pickle.dump(feat_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

fip_scaler = StandardScaler()
fip_scaler.fit(fip_data)

with open(file=f'{curfolder}\\fip_scaler.pickle', mode='wb') as f:
    pickle.dump(fip_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

fip_avg_scaler = StandardScaler()
fip_avg_scaler.fit(fip_avg_data)

with open(file=f'{curfolder}\\fip_avg_scaler.pickle', mode='wb') as f:
    pickle.dump(fip_avg_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)