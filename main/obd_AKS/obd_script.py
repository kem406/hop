## imports
import numpy as np
import main.obd.obd as ob
import imageio
import matplotlib.pyplot as plt
plt.style.use('dark_background')

## parameters
sf = np.array([50, 50])       # size of the PSF
maxiter = [50, 2]    # number of iterations for f and x
n = 40              # number of images
clipping = np.inf      # maximally acceptable pixel (for saturation correction)
srf = 1.0           # superresolution factor

# how are the filenames generated?
imagepath = 'obd/data/epsilon_lyrae/'

def y_fname(i):
    return imagepath+"{0:08d}".format(i)+'.png'

test = imageio.imread(y_fname(10))

# intially there is no x
x = np.array([])

for i in range(1,40):
  # load the next observed image
  fname = y_fname(i)
  print('Processing {}'.format(fname))
  y = imageio.imread(fname)
  y = y[:,20:132,2].astype(np.float64)   # use only first color channel

  ##### THE MAIN WORK HORSE #####
  x, f = ob.obd(x, y, sf, maxiter, clipping, srf)

  fig, ax = plt.subplots(1,3, figsize=(24., 8.))
  ax[0].imshow(y, origin='lower')
  ax[1].imshow(f, origin='lower')
  ax[2].imshow(x, origin='lower')
  plt.show()
