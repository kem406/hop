## imports
import numpy as np
import main.obd.obd as ob
import imageio
import matplotlib.pyplot as plt

imagepath = 'transit_mocks/data1/super_resolution/'
imagepath0= 'transit_mocks/data1/'
def y_fname(i):
    return imagepath+'super_resolution_'+str(i)+'.tiff'
def gt_y_fname(i):
    return imagepath0+'transit_test_gt_'+str(i)+'.tiff'

#test = imageio.imread(y_fname(1))
#print(np.sum(test))

plot0=np.array([np.sum(imageio.imread(gt_y_fname(0)))])
for i in range (1,101):
    print(i)
    plot=np.concatenate((plot0, np.array([np.sum(imageio.imread(gt_y_fname(i)))])), axis=0)
print(plot0)

plot=np.array([np.sum(imageio.imread(y_fname(0)))])
for i in range (1,101):
    plot=np.concatenate((plot, np.array([np.sum(imageio.imread(y_fname(i)))])), axis=0)
print(plot)
