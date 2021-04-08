## imports
import numpy as np
import main.obd_AKS.obd as ob
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

## parameters
sf = np.array([50, 50])       # size of the PSF
maxiter = [40, 1]    # number of iterations for f and x
n = 40              # number of images
clipping = np.inf      # maximally acceptable pixel (for saturation correction)
srf = 1.0           # superresolution factor

# how are the filenames generated?
imagepath = 'transit_mocks/data1/'

def gt_y_fname(i):
    return imagepath+'transit_test_gt_'+str(i)+'.tiff'
#transit_test_0_slice_0
test = imageio.imread(gt_y_fname(1))

def y_fname(i,j):
    return imagepath+'transit_test_'+str(i)+'_slice_'+str(j)+'.tiff'
#transit_test_0_slice_0
test = imageio.imread(y_fname(1,1))

def gt_y_fname_nocurve(i):
    return imagepath+'NOTRANSIT_test_gt_'+str(i)+'.tiff'
#transit_test_0_slice_0

def y_fname_nocurve(i,j):
    return imagepath+'NOTRANSIT_test_'+str(i)+'_slice_'+str(j)+'.tiff'

# intially there is no x
x = np.array([])

# OBD for transit
for i in tqdm(range(0,101)):
    for j in range(0,40):
        # load the next observed image
        fname = y_fname(i,j)
        y = imageio.imread(fname)
        #y = y[:,:,1].astype(np.float64)   # use only first color channel
        #print('Processing {}'.format(fname))
        x, f = ob.obd(x, y, sf, maxiter)
        #fig, ax = plt.subplots(1,3, figsize=(24., 8.))
        #ax[0].imshow(y, origin='lower')
        #ax[1].imshow(f, origin='lower')
        #ax[2].imshow(x, origin='lower')
        #plt.show()
    #x=x.astype(np.uint8)
    #x=convert(x, 0, 255, np.uint8)
    imageio.imwrite(imagepath+'super_resolution/super_resolution_'+str(i)+'.tiff', x)

# OBD for no transit
for i in tqdm(range(0,100)):
    for j in range(0,40):
        # load the next observed image
        fname = y_fname_nocurve(i,j)
        y = imageio.imread(fname)
        #y = y[:,:,1].astype(np.float64)   # use only first color channel
        #print('Processing {}'.format(fname))
        x, f = ob.obd(x, y, sf, maxiter)
        #fig, ax = plt.subplots(1,3, figsize=(24., 8.))
        #ax[0].imshow(y, origin='lower')
        #ax[1].imshow(f, origin='lower')
        #ax[2].imshow(x, origin='lower')
        #plt.show()
    #x=x.astype(np.uint8)
    #x=convert(x, 0, 255, np.uint8)
    imageio.imwrite(imagepath+'super_resolution/NOTRANSIT_super_resolution_'+str(i)+'.tiff', x)
