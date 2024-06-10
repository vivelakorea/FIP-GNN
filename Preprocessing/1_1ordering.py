import shutil
import os


def arrange_txt(texture, start, end):
    # texture = 90
    # start = 22
    # end = 43

    curfolder = os.getcwd()
    foldername = f'{curfolder}\\{texture}'

    for i in range(end-start+1):

        shutil.copy(f'{foldername}\\{start}-{end}\\grainID_{i}.txt', f'{foldername}\\grainID_{i+start}.txt')
        shutil.copy(f'{foldername}\\{start}-{end}\\orientations_{i}.txt', f'{foldername}\\orientations_{i+start}.txt')

arrange_txt(90, 0, 21)
arrange_txt(90, 22, 43)
arrange_txt(90, 44, 199)

arrange_txt(160, 0, 3)
arrange_txt(160, 4, 7)
arrange_txt(160, 8, 99)
