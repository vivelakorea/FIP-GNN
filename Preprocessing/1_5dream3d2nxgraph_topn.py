import h5py
import numpy as np
import os
import networkx as nx
import multiprocessing

def __get_orientation_matrix(euler_angles):
    euler1, euler2, euler3 = euler_angles

    COS = np.cos
    SIN = np.sin

    orientation_matrix = np.array([
        [COS(euler1)*COS(euler3)-SIN(euler1)*SIN(euler3)*COS(euler2), SIN(euler1)*COS(euler3)+COS(euler1)*SIN(euler3)*COS(euler2), SIN(euler2)*SIN(euler3)],
        [-COS(euler1)*SIN(euler3)-SIN(euler1)*COS(euler3)*COS(euler2), -SIN(euler1)*SIN(euler3)+COS(euler1)*COS(euler3)*COS(euler2), SIN(euler2)*COS(euler3)],
        [SIN(euler1)*SIN(euler2), -COS(euler1)*SIN(euler2), COS(euler2)]
    ])

    return orientation_matrix
def __get_fcc_schmids(euler_angles, loading_direction): 
    '''
    Calculate all euler angles of fcc
    
    euler_angles : Bunge Euler angles in randian
    loading_direction : ex) [1, 0, 0]
    '''
    
    schmid_factors = []

    crystal_info = [
        {'hkl': [1.,1.,1.], 'uvw': [0.,1.,-1.]},
        {'hkl': [1.,1.,1.], 'uvw': [-1.,0.,1.]},
        {'hkl': [1.,1.,1.], 'uvw': [1.,-1.,0.]},

        {'hkl': [-1.,-1.,1.], 'uvw': [0.,-1.,-1.]},
        {'hkl': [-1.,-1.,1.], 'uvw': [1.,0.,1.]},
        {'hkl': [-1.,-1.,1.], 'uvw': [-1.,1.,0.]},

        {'hkl': [1.,-1.,-1.], 'uvw': [0.,-1.,1.]},
        {'hkl': [1.,-1.,-1.], 'uvw': [-1.,0.,-1.]},
        {'hkl': [1.,-1.,-1.], 'uvw': [1.,1.,0.]},

        {'hkl': [-1.,1.,-1.], 'uvw': [0.,1.,1.]},
        {'hkl': [-1.,1.,-1.], 'uvw': [1.,0.,-1.]},
        {'hkl': [-1.,1.,-1.], 'uvw': [-1.,-1.,0.]},
    ]

    orientation_matrix = __get_orientation_matrix(euler_angles)

    crs_load_dir = np.matmul(orientation_matrix, np.array(loading_direction))
    xb, yb, zb = crs_load_dir
    length = np.sqrt(xb**2. + yb**2. + zb**2.)

    for cry_sys in crystal_info:
        hkl = np.array(cry_sys['hkl'])
        uvw = np.array(cry_sys['uvw'])
        
        cosphi = np.dot(hkl, crs_load_dir)/(np.sqrt(3)*length)
        coslam = np.dot(uvw, crs_load_dir)/(np.sqrt(2)*length)

        schmid = abs(cosphi*coslam)
        schmid_factors.append(schmid)

    return schmid_factors



################################################################################################################################################################



def write_nx_graph(texture, ith_sve):
    # texture = 30
    # ith_sve = 0
    curfolder = os.getcwd()

    foldername = f'{curfolder}\\{texture}'
    '''Get neighbors of grain, using hdf5 file(*.dream3d)'''

    f = h5py.File(f'{foldername}\\Output_FakeMatl_{ith_sve}_duplicated.dream3d', 'r')

    eulerAngles = np.array(f['DataContainers']['SyntheticVolumeDataContainer']['CellFeatureData']['EulerAngles'])
    eulerAngles = eulerAngles[:,::-1]

    numNeighbors = np.array(f['DataContainers']['SyntheticVolumeDataContainer']['CellFeatureData']['NumNeighbors'])
    numNeighbors = numNeighbors[1:,0]

    neighborList = np.array(f['DataContainers']['SyntheticVolumeDataContainer']['CellFeatureData']['NeighborList'])

    nbr_dict = {}

    feat = 1
    i = 0
    for numNeighbor in numNeighbors:
        nbr_dict[int(feat)] = list(map(int, neighborList[i:i+numNeighbor]))
        i += numNeighbor
        feat += 1

    G = nx.Graph()
    for feat, ori in enumerate(eulerAngles):
        if feat == 0: continue

        schmids = __get_fcc_schmids(ori, loading_direction=[1, 0, 0])
        sorted_schmid = sorted(schmids, reverse=True)
        G.add_nodes_from([(feat, {"x": np.hstack([sorted_schmid[:3]])})])

    for feat, nbrs in nbr_dict.items():
        for nbr in nbrs:
            G.add_edge(feat, nbr, e=0)

    nx.write_gpickle(G, f'{foldername}\\sve_top3_{ith_sve}')

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

def func(ith_sve):
    
    ############## change here ##############
    texture = 30
    #########################################

    write_nx_graph(texture,ith_sve)

if __name__ == '__main__':

    ############### change here ###############
    numSVEs = 200
    ############################################
    
    pool_obj = multiprocessing.Pool(61)
    pool_obj.map(func, list(range(numSVEs)))