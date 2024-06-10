import os
import numpy as np
import h5py
import multiprocessing

def write_fip_table(fiptable_dir, fiphdf5_dir, texture, ith_sve):
    
    # https://materialscommons.org/public/datasets/248/overview

    hdf5s = {
        30: 
            {
            'fname': fiphdf5_dir + '\\30_voxels_per_side_200_samples_275_avg_num_grains.hdf5',
            'featname': '275_grain_data'
            },
        45: 
            {
            'fname': fiphdf5_dir + '\\45_voxels_per_side_200_samples_950_avg_num_grains.hdf5', 
            'featname': '950_grain_data'
            },
        90: 
            {
            'fname': fiphdf5_dir + '\\90_voxels_per_side_200_samples_7500_avg_num_grains.hdf5', 
            'featname': '7500_grain_data'
            },
        160:
            {
            'fname': fiphdf5_dir + '\\160_voxels_per_side_100_samples_41000_avg_num_grains.hdf5', 
            'featname': '41000_grain_data'
            },
        200:
            {
            'fname': fiphdf5_dir + '\\200_voxels_per_side_4_samples_80000_avg_num_grains.hdf5', 
            'featname': '80000_grain_data'
            },
        250:
            {
            'fname': fiphdf5_dir + '\\250_voxels_per_side_2_samples_160000_avg_num_grains.hdf5', 
            'featname': '160000_grain_data'
            },
    }

    hdf5 = hdf5s[texture]['fname']
    featname = hdf5s[texture]['featname']
    elem_per_side = int(texture)

    f = h5py.File(hdf5, 'r')
    data = np.array(f.get(featname))
    f.close()

    elemFIPs = data[ith_sve, 1, :].reshape((elem_per_side, elem_per_side, elem_per_side))
    grainIDs = data[ith_sve, 2, :].reshape((elem_per_side, elem_per_side, elem_per_side))

    fips = {} # grainID: elemFIP
    for i in range(elem_per_side):
        for j in range(elem_per_side):
            for k in range(elem_per_side):
                grainID = int(grainIDs[i, j, k])
                elemFIP = elemFIPs[i, j, k]
                if grainID not in fips:
                    fips[grainID] = elemFIP
                else: # test if all element in same grain has same fip
                    assert(elemFIP == fips[grainID])

    f = open(f'{fiptable_dir}\\fip_{ith_sve}.csv', 'w')
    
    numGrains = len(fips.keys())
    
    for i in range(1, numGrains+1):
        f.write(f'{i},{fips[i]}\n')

    f.close()

######################################################################################################################################
def write_fip_table_avg(fiptable_dir, fiphdf5_dir, texture, ith_sve):
    
    # https://materialscommons.org/public/datasets/248/overview

    hdf5s = {
        30: 
            {
            'fname': fiphdf5_dir + '\\30_voxels_per_side_200_samples_275_avg_num_grains.hdf5',
            'featname': '275_grain_data'
            },
        45: 
            {
            'fname': fiphdf5_dir + '\\45_voxels_per_side_200_samples_950_avg_num_grains.hdf5', 
            'featname': '950_grain_data'
            },
        90: 
            {
            'fname': fiphdf5_dir + '\\90_voxels_per_side_200_samples_7500_avg_num_grains.hdf5', 
            'featname': '7500_grain_data'
            },
        160:
            {
            'fname': fiphdf5_dir + '\\160_voxels_per_side_100_samples_41000_avg_num_grains.hdf5', 
            'featname': '41000_grain_data'
            },
        200:
            {
            'fname': fiphdf5_dir + '\\200_voxels_per_side_4_samples_80000_avg_num_grains.hdf5', 
            'featname': '80000_grain_data'
            },
        250:
            {
            'fname': fiphdf5_dir + '\\250_voxels_per_side_2_samples_160000_avg_num_grains.hdf5', 
            'featname': '160000_grain_data'
            },
    }

    hdf5 = hdf5s[texture]['fname']
    featname = hdf5s[texture]['featname']
    elem_per_side = int(texture)

    f = h5py.File(hdf5, 'r')
    data = np.array(f.get(featname))
    f.close()

    elemFIPs = data[ith_sve, 0, :].reshape((elem_per_side, elem_per_side, elem_per_side))
    grainIDs = data[ith_sve, 2, :].reshape((elem_per_side, elem_per_side, elem_per_side))

    fips = {} # grainID: elemFIP
    for i in range(elem_per_side):
        for j in range(elem_per_side):
            for k in range(elem_per_side):
                grainID = int(grainIDs[i, j, k])
                elemFIP = elemFIPs[i, j, k]
                if grainID not in fips:
                    fips[grainID] = [elemFIP]
                else: 
                    fips[grainID].append(elemFIP)
    
    avg_fips = {}
    for grainID in fips:
        avg_fips[grainID] = np.average(fips[grainID])

    f = open(f'{fiptable_dir}\\fip_{ith_sve}_avg.csv', 'w')
    
    numGrains = len(avg_fips.keys())
    
    for i in range(1, numGrains+1):
        f.write(f'{i},{avg_fips[i]}\n')

    f.close()

######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

def func(ith_sve):
    ############## change here ##############
    texture = 160
    #########################################

    curfolder = os.getcwd()
    inputFolder = f'{curfolder}\\FIP'
    outputFolder = f'{curfolder}\\{texture}'
    
    write_fip_table(outputFolder, inputFolder, texture, ith_sve)
    write_fip_table_avg(outputFolder, inputFolder, texture, ith_sve)

if __name__ == '__main__':

    ############### change here ###############
    numSVEs = 100
    ############################################
    
    # pool_obj = multiprocessing.Pool(61)
    # pool_obj.map(func, list(range(numSVEs)))
    for i in range(numSVEs):
        func(i)
        print(i,numSVEs)
