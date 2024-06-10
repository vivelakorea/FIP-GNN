import numpy as np
import h5py
import os
import multiprocessing

def duplicategrainIDs(texture, i):
# texture = 30
# numSVEs = 200

    curfolder = os.getcwd()
    foldername = f'{curfolder}\\{texture}'

    # read ori
    # i = 0
    dream3d_fname = f'{foldername}\\Output_FakeMatl_{i}.dream3d'
    f = h5py.File(dream3d_fname, 'r')

    featureIds = np.array(f['DataContainers']['SyntheticVolumeDataContainer']['CellData']['FeatureIds'])
    featureIds = featureIds[:,:,:,0]

    # duplicate featureIds
    dup_featurIds = np.tile(featureIds, (3,3,3))
    flatten_dup_featureIds = dup_featurIds.reshape((texture**2)*9, texture*3)
    txt_flatten_dup_featureIds = '\n'.join(' '.join('%d' %x for x in y) for y in flatten_dup_featureIds)+'\n'

    # write duplicated featureIds
    f = open(f'{foldername}\\grainID_{i}_duplicated.txt', 'w')
    f.write(f'''# object 1 are the regular positions. The grid is {texture*3} {texture*3} {texture*3}. The origin is
# at [0 0 0], and the deltas are 1 in the first and third dimensions, and
# 2 in the second dimension
#
object 1 class gridpositions counts {texture*3} {texture*3} {texture*3}
origin 0 0 0
delta  1 0 0
delta  0 1 0
delta  0 0 1
#
# object 2 are the regular connections
#
object 2 class gridconnections counts {texture*3} {texture*3} {texture*3}
#
# object 3 are the data, which are in a one-to-one correspondence with
# the positions ("dep" on positions). The positions increment in the order
# "last index varies fastest", i.e. (x0, y0, z0), (x0, y0, z1), (x0, y0, z2),
# (x0, y1, z0), etc.
#
object 3 class array type int rank 0 items {(texture*3)**3} data follows
'''
    )

    f.write(txt_flatten_dup_featureIds)

    f.write('''attribute "dep" string "positions"
#
# A field is created with three components: "positions", "connections",
# and "data"
object "regular positions regular connections" class field
component  "positions"    value 1
component  "connections"  value 2
component  "data"         value 3
#
end'''
    )

    f.close()

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

def func(i):
    
    ############## change here ##############
    texture = 160
    #########################################

    duplicategrainIDs(texture,i)


if __name__ == '__main__':

    ############### change here ###############
    numSVEs = 100
    ############################################
    
    # pool_obj = multiprocessing.Pool(61)
    # pool_obj.map(func, list(range(numSVEs)))
    for i in range(numSVEs):
        func(i)
        print(i,numSVEs)
    