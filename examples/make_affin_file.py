import h5py
import affinities
import numpy as np

f = h5py.File("sample_A_20160501.hdf", "r")
segmentation=f["volumes"]["labels"]["neuron_ids"].value



affinities=affinities.get_affins(segmentation)
affinities=np.asarray(affinities,dtype=np.int)

np.save("affinities",affinities)

print("DONE")
