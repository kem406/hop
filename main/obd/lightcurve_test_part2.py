## imports
import numpy as np
import main.obd_AKS.obd as ob
import imageio

#Plotting Packages
import matplotlib as mpl
import matplotlib.cm as cmplt
import matplotlib.pyplot as plt
from matplotlib import rcParams

imagepath = 'transit_mocks/data1/super_resolution/'
imagepath0= 'transit_mocks/data1/'

def y_fname(i):
    return imagepath+'super_resolution_'+str(i)+'.tiff'
def gt_y_fname(i):
    return imagepath0+'transit_test_gt_'+str(i)+'.tiff'

test = imageio.imread(gt_y_fname(1))
#print(np.sum(test))

plot0_y=np.array([np.sum(imageio.imread(gt_y_fname(0)))])
plot0_x=np.array([0])
for i in range (1,101):
    #print(np.sum(imageio.imread(gt_y_fname(i))))
    plot0_y=np.concatenate((plot0_y, np.array([np.sum(imageio.imread(gt_y_fname(i)))])), axis=0)
    plot0_x=np.concatenate((plot0_x, np.array([i])), axis=0)
#print(plot0_x)
#print(plot0_y)
plt.scatter(plot0_x, plot0_y)

plot_y=np.array([np.sum(imageio.imread(y_fname(0)))])
for i in range (1,101):
    plot_y=np.concatenate((plot_y, np.array([np.sum(imageio.imread(y_fname(i)))])), axis=0)
#print(plot_y)
plt.scatter(plot0_x, plot_y)
