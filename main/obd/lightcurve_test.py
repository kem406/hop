## imports
import numpy as np
import main.obd.obd as ob
import imageio
import matplotlib.pyplot as plt

## parameters
sf = np.array([50, 50])       # size of the PSF
maxiter = [50, 2]    # number of iterations for f and x
n = 40              # number of images
clipping = np.inf      # maximally acceptable pixel (for saturation correction)
srf = 1.0           # superresolution factor

# how are the filenames generated?
imagepath = 'transit_mocks/data/'

def y_fname(i,j):
    return imagepath+'transit_test_'+str(i)+'_slice_'+str(j)+'.png'
#transit_test_0_slice_0
test = imageio.imread(y_fname(1,1))

# intially there is no x
x = np.array([])

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

for i in range(0,101):
    for j in range(0,40):
        # load the next observed image
        fname = y_fname(i,j)
        y = imageio.imread(fname)
        #y = y[:,:,1].astype(np.float64)   # use only first color channel
        #print('Processing {}'.format(fname))
        x, f = ob.obd(x, y, sf, maxiter, clipping, srf)
        #fig, ax = plt.subplots(1,3, figsize=(24., 8.))
        #ax[0].imshow(y, origin='lower')
        #ax[1].imshow(f, origin='lower')
        #ax[2].imshow(x, origin='lower')
        #plt.show()
    x=convert(x, 0, 255, np.uint8)
    imageio.imwrite(imagepath+'super_resolution/super_resolution_'+str(i)+'.png', x)
