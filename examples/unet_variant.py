from elektronn2 import neuromancer as nm
import elektronn2.malis as malis
import code
from elektronn2.utils import h5load
import h5py

                    ###################
                    #   GET  DATA    #
                    ###################



f = h5py.File("/Users/liammcgoldrick/Code/ELEKTRONN2/examples/sample_A_20160501.hdf", "r")
raw_data=f["volumes"]["raw"].value
segmentation=f["volumes"]["labels"]["neuron_ids"].value

affins=get_affins(segmentation)





                    ###################
                    #   BUILD GRAPH   #
                    ###################


in_sh = (None,1,128,128,16)

inp = nm.Input(in_sh, 'b,f,x,y,z', name='raw')  # high res

# Convolution, downsampling of intermediate features

conv0  = nm.Conv(inp,  8,  (15,15,15),(1,1,1),"same",name="c0")
down0  = nm.Pool(conv0,(4,4,1),mode="max",name="d0")#full

conv1  = nm.Conv(down0,  32,  (15,15,15),(1,1,1),"same",name="c1")
down1  = nm.Pool(conv1,(2,2,1),mode="max",name="d1")#high

conv2  = nm.Conv(down1,  64,  (15,15,15),(1,1,1),"same",name="c2")#mid
down2  = nm.Pool(conv2,(2,2,1),mode="max",name="d2")#high

conv3  = nm.Conv(down2,  64,  (15,15,15),(1,1,1),"same",name="c3")#low


#merge C2 and C3 and convolve
mrg0   = nm.UpConvMerge(conv2, conv3, 64,name="m0",merge_mode="add")
mconv0=nm.Conv(mrg0,32,(15,15,15),(1,1,1),"same",name="mc0")

#merge mc0 and c1 and convolve
mrg1   = nm.UpConvMerge(mconv0, conv1, 32,name="m1",merge_mode="add")
mconv1 = nm.Conv(mrg1,8,(15,15,15),(1,1,1),"same",name="mc1")

#merge mc1 and c0 and convolve
mrg2   = nm.UpConvMerge(mconv1, conv0, 8,name="m2",merge_mode="add")
mconv2 = nm.Conv(mrg2,6,(15,15,15),(1,1,1),"same",name="mc2")

affins=nm.Input((None,6,128,128,16), 'b,f,x,y,z', name='affins')

loss_node = nm.loss.BinaryNLL(mconv2,affins)
loss=nm.AggregateLoss(loss_node)


model = nm.model_manager.getmodel()
model.designate_nodes(
    input_node=inp,
    target_node=affins,
    loss_node=loss,
    prediction_node=mconv2
)

                    ###################
                    #    RUN  GRAPH   #
                    ###################




#test to make sure everything works
model.test_run_prediction()


#model.trainingstep(data= raw_data ,target= ,optimiser="Adam")

