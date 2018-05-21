import numpy as np
import code
import time

def get_affins(arr):
    #takes in (z,x,y)
    arr = np.asarray(arr)
    print("SHAPE: ",np.shape(arr))

    print("GETTING AFFINITIES...")

    time_s=time.time()


    affins=np.zeros((3,np.shape(arr)[0],np.shape(arr)[1],np.shape(arr)[2]))



    #check up
    for z in range(0,np.shape(arr)[0]-1):
        affins[0,z,:,:]= (arr[z,:,:]==arr[z+1,:,:])
    print("     PROCESSED Z AFFINITIES")
    print("     TIME", int(time.time()-time_s),"s")

    #check right
    for x in range(0, np.shape(arr)[1] - 1):
        affins[1,:,x,:] = (arr[:,x,:] == arr[:,x+1,:])
    print("     PROCESSED X AFFINITIES")
    print("     TIME", int(time.time() - time_s), "s")

    #check in
    for y in range(0, np.shape(arr)[2] - 1):
        affins[2,:,:,y] = (arr[:,:,y] == arr[:,:,y+1])
    print("     PROCESSED Y AFFINITIES")
    print("     TIME", int(time.time() - time_s), "s")

    print("DONE")

    affins=np.asarray(affins,dtype=np.int)

    return affins

