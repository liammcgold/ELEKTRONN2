import numpy as np
import code

def get_affins(arr):
    #takes in (z,x,y)

    arr=np.asarray(arr)

    affins=np.zeros((3,np.shape(arr)[0],np.shape(arr)[1],np.shape(arr)[2]))

    #comp up
    for z in range(0,np.shape(arr)[0]-1):
        affins[0,z,:,:]= (arr[z,:,:]==arr[z+1,:,:])

    #check right
    for x in range(0, np.shape(arr)[1] - 1):
        affins[1,:,x,:] = (arr[:,x,:] == arr[:,x+1,:])

    #check in
    for y in range(0, np.shape(arr)[2] - 1):
        affins[2,:,:,y] = (arr[:,:,y] == arr[:,:,y+1])

    return affins

