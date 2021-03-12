## imports
import numpy as np
import obd_AKS.obd as ob
import debayer as db
import imageio
import matplotlib.pyplot as plt
import glob
plt.style.use('dark_background')

## parameters
sf = np.array([64,64])       # size of the PSF
maxiter = [50, 1]    # number of iterations for f and x
n = 10              # number of images
clipping = np.inf      # maximally acceptable pixel (for saturation correction)
srf = 1.0           # superresolution factor

# how are the filenames generated?
imagepath = '../testdata/'

def y_fname(i):
    filenames = np.sort(glob.glob(imagepath+'*.fits'))
    return filenames[i]

# intially there is no x
x = np.array([])

for i in range(0,n):
  # load the next observed image
  fname = y_fname(i)
  print('Processing {}'.format(fname))
  y = db.debayer(fname)
  y = y[207:335,298:426,1].astype(np.float64)   # use only green color channel

  ##### THE MAIN WORK HORSE #####
  x, f = ob.obd(x, y, sf, maxiter)

  fig, ax = plt.subplots(1,3, figsize=(24., 8.))
  ax[0].imshow(y, origin='lower')
  ax[1].imshow(f, origin='upper')
  ax[2].imshow(x, origin='lower')
  plt.show()
