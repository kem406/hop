## imports
import numpy as np
import main.obd.obd as ob
import imageio
import matplotlib.pyplot as plt

## parameters
sf = np.array([50, 50])       # size of the PSF
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

for i in range(1,2):
  # load the next observed image
  fname = y_fname(i)
  print('Processing {}'.format(fname))
  y = imageio.imread(fname)
  y = y[:,:,1].astype(np.float64)   # use only first color channel

  ##### THE MAIN WORK HORSE #####
  x, f = ob.obd(x, y, sf, maxiter, clipping, srf)

print('Done! The result is in variable "x"')

np.count_nonzero(f)

plt.imshow(y)

plt.imshow(f)

temp = ob.cnv2tp(f, y, srf)

plt.imshow(ob.setZero(temp))

plt.imshow(x)


f = np.linalg.norm(np.ndarray.flatten(y)) / np.linalg.norm(np.ndarray.flatten(x))
f
x.shape
y.shape
np.ndarray.flatten(y)
np.ndarray.flatten(x)

f = np.linalg.norm(np.ndarray.flatten(y)) / np.linalg.norm(np.ndarray.flatten(x))
f = f * np.ones(sf) / np.sqrt(np.prod(sf, axis=0))

plt.imshow(f)

f1 = ob.obd_update(f, x, y, maxiter[0], clipping, srf)

plt.imshow(f1)

np.count_nonzero(f1)


type(sf[0])

sf

y.shape
