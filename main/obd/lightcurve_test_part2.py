## imports
import numpy as np
import main.obd.obd as ob
import imageio
import matplotlib.pyplot as plt

imagepath = 'transit_mocks/data/super_resolution/'

def y_fname(i):
    return imagepath+'super_resolution_'+str(i)+'.png'

#test = imageio.imread(y_fname(1))
#print(np.sum(test))

plot=np.array([np.sum(imageio.imread(y_fname(0)))])
for i in range (1,101):
    plot=np.concatenate((plot, np.array([np.sum(imageio.imread(y_fname(i)))])), axis=0)
print(plot)
