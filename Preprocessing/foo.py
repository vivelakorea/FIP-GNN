import pickle
import os
import math

path = os.getcwd()


fname = f'{path}\\fip_avg_scaler.pickle'

with open(file=fname, mode='rb') as f:
    fip_scaler = pickle.load(f)

# print(dir(fip_scaler))
print(fip_scaler.mean_)
print(math.sqrt(fip_scaler.var_))
print()