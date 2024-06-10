import os
import json
import math
import multiprocessing

def retxt2dream3d(texture, i):
    # texture = 30
    # numSVEs = 200

    curfolder = os.getcwd()
    foldername = f'{curfolder}\\{texture}'

    # for i in range(numSVEs):
        
    grainIDFile = f'{foldername}\\grainID_{i}_duplicated.txt'
    orientationsFile_in = f'{foldername}\\orientations_{i}.txt'
    

    # modify orientationsFile_in to not have grainID
    orientationsFile_out = f'{foldername}\\orientations_{i}_out.txt'
    
    # f = open(orientationsFile_in, 'r'); lines = f.readlines(); f.close()

    # f = open(orientationsFile_out, 'w')
    # for line in lines:
    #     tmp = list(map(float,(line.split(' '))))
    #     grainId, rcomp1, rcomp2, rcomp3, phase = tmp[:4]
    #     rodrigues4 = math.sqrt(rcomp1**2. + rcomp2**2. + rcomp3**2.)
    #     rodrigues1 = -rcomp1/rodrigues4 if rodrigues4 != 0 else 0
    #     rodrigues2 = rcomp2/rodrigues4 if rodrigues4 != 0 else 0
    #     rodrigues3 = -rcomp3/rodrigues4 if rodrigues4 != 0 else 0
    #     f.write(f'{rodrigues1} {rodrigues2} {rodrigues3} {rodrigues4}\n')
    # f.close()

    # read number of grains
    f = open(orientationsFile_in, 'r'); lines = f.readlines(); f.close()
    tmp = list(map(float,(lines[-1].split(' '))))
    n = int(tmp[0])

    # modify Dream3D pipeline
    outputFile = f'{foldername}\\Output_FakeMatl_{i}_duplicated.dream3d'

    data = json.load(open(f'{curfolder}\\reread_using_grain_and_ori2.json'))
    data['00']['InputFile'] = grainIDFile
    data['03']['TupleDimensions']['Table Data'] = [[n+1]]
    data['05']['InputFile'] = orientationsFile_out
    data['15']['OutputFile'] = outputFile

    modifiedPipeline = f"{foldername}\\tmp_{i}.json"
    with open(modifiedPipeline, "w") as outfile:
        json.dump(data, outfile)

    # run the pipeline
    dream3dEXE = "C:\\Users\\Gyu-Jang Sim\\Desktop\\DREAM3D-6.5.171-Win64\\PipelineRunner.exe"
    os.system(f'echo | \"{dream3dEXE}\" -p \"{modifiedPipeline}\"')


# retxt2dream3d(30,200)
# retxt2dream3d(45,200)
# retxt2dream3d(90,200)
# retxt2dream3d(160,100)
# retxt2dream3d(200,4)
# retxt2dream3d(250,2)
        

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

def func(i):
    
    ############## change here ##############
    texture = 160
    #########################################

    retxt2dream3d(texture,i)


if __name__ == '__main__':

    ############### change here ###############
    numSVEs = 100
    ############################################
    
    # pool_obj = multiprocessing.Pool(61)
    # pool_obj.map(func, list(range(numSVEs)))

    for i in range(numSVEs):
        func(i)
        print(i,numSVEs)
    