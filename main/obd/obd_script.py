## imports
import numpy as np
import main.obd.obd as obd
import imageio
import matplotlib.pyplot as plt

## parameters
sf = [50, 50]        # size of the PSF
maxiter = [50, 1]    # number of iterations for f and x
n = 40              # number of images
clipping = np.inf      # maximally acceptable pixel (for saturation correction)
srf = 1.0           # superresolution factor

# how are the filenames generated?
imagepath = 'obd/data/epsilon_lyrae/'

def y_fname(i):
    return imagepath+"{0:08d}".format(i)+'.png'

test = imageio.imread(y_fname(10))

# intially there is no x
x = []

for i in range(1,n+1):
  # load the next observed image
  fname = y_fname(i)
  print('Processing {}'.format(fname))
  y = imageio.imread(fname)
  y = y[:,:,1].astype(np.float64)   # use only first color channel

  ##### THE MAIN WORK HORSE #####
  x, f = ob.obd(x, y, sf, maxiter, clipping, srf)

print('Done! The result is in variable "x"');
