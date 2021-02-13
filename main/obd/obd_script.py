## imports
import numpy as np
import main.obd.obd as ob
import imageio
import matplotlib.pyplot as plt

import pdb

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
  y = y[:,:,1].astype(np.float64)   # use only first color channel

  ##### THE MAIN WORK HORSE #####
  x, f = ob.obd(x, y, sf, maxiter, clipping, srf)

  fig, ax = plt.subplots(1,3, figsize=(24., 8.))
  ax[0].imshow(y, origin='lower')
  ax[1].imshow(f, origin='lower')
  ax[2].imshow(x, origin='lower')
  plt.show()

1+1
plt.imshow(x)

x.shape

plt.imshow(y)

y.shape

plt.imshow(f)
f.shape
f

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

f

f1 = ob.obd_update(f, x, y, 2, clipping, srf)

f1

plt.imshow(f1[0])

plt.imshow(f1[1])

np.count_nonzero(f1[1])

plt.imshow(f1[2])

np.count_nonzero(f1[2])

np.count_nonzero(f1[3])

f1[3]

plt.imshow(np.real(f1[4]))

x.shape
y.shape

ob.cnv2tp(x, y, srf)


type(sf[0])

sf

y.shape

y
10*y

sy
len(x)
sy
np.shape(np.array([]))
np.shape(x)

np.shape(np.array([])) == (0,)

y

(1,3)-(2,4)
np.array(np.shape(x))-np.array(np.shape(y))

A = np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]])
A
B = np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]])

np.divide(A,B)

np.array(np.shape(np.array([])))[0]


f = ob.obd_update(f, x, y, maxiter[0], clipping, srf)

plt.imshow(f)

f

sx = np.array(x.shape)
sy = np.array(y.shape)
sf = sx - sy
sf

test1 = np.fft.ifft2(np.multiply(np.fft.fft2(x), np.fft.fft2(ob.cnv2pad(y, sf))))
test2 = ob.cnv2slice(test1, slice(0, sf[0]), slice(0, sf[1]))
plt.imshow(np.abs(test1))

plt.imshow(np.real(test2))

test2.shape

np.count_nonzero(test1)

np.shape(np.fft.fft2(x, s=sf))

np.shape(np.fft.fft2(x))

np.shape(np.fft.fft2(ob.cnv2pad(y, sf)))


f3 = ob.obd_update(f, x, y, maxiter[0], clipping, srf)

plt.imshow(f3[0])

plt.imshow(f3[1])

plt.imshow(f3[2])

plt.imshow(f3[3])

plt.imshow(np.real(f3[4]))

f3
np.count_nonzero(f3[2])


test1 = np.fft.ifft2(np.multiply(np.fft.fft2(x), np.fft.fft2(ob.cnv2pad(y, sf))))
test2 = ob.cnv2slice(test1, slice(0, sf[0]), slice(0, sf[1]))

plt.imshow(np.real(test2))

f = np.fft.ifft2(np.multiply(np.fft.fft2(x, s=sf), np.fft.fft2(ob.cnv2pad(y, sf),s=sf)))

plt.imshow(x)

plt.imshow(np.real(f))

f = cnv2slice(np.real(f), slice(0, sf[0]), slice(0, sf[1]))

ob.cnv2tp(x, y, srf)
