import numpy as np


def get_affins(arr):
    affins=np.zeros((3,np.shape(arr)[0],np.shape(arr)[1],np.shape(arr)[2]))
    for z in range(0,np.shape(arr)[0]):
        for x in range(0,np.shape(arr)[1]):
            for y in range(0, np.shape(arr)[2]):
                #find up affin
                if(z!=np.shape(arr)[0]-1):
                    if(arr[z][x][y]==arr[z+1][x][y]):
                        affins[0][z][x][y]=1
                    else:
                        affins[0][z][x][y] = 0
                else:
                    affins[0][z][x][y] = 0

                # find inward affin
                if (y != np.shape(arr)[2] - 1):
                    if (arr[z][x][y] == arr[z][x][y + 1]):
                        affins[1][z][x][y] = 1
                    else:
                        affins[1][z][x][y] = 0
                else:
                    affins[1][z][x][y] = 0


                # find right affin
                if (x != np.shape(arr)[1]-1 ):
                    if (arr[z][x][y] == arr[z][x+1][y]):
                        affins[2][z][x][y] = 1
                    else:
                        affins[2][z][x][y] = 0
                else:
                    affins[2][z][x][y] = 0


    return affins
