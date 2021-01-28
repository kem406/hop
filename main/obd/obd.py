import numpy as np
import pdb
import matplotlib.pyplot as plt

def obd(x, y, sf, maxiter, clipping=np.inf, srf=1):

  # find dimensions of y
  sy = np.array(np.shape(y))

  # multiply sy and sf by srf and round down to nearest integer if srf above 1
  if (srf > 1):
    sy = np.floor(srf * sy)
    sf = np.floor(srf * sf)

  # srf below 1 is invalid by def. since it is ratio of higher freq to lower freq
  elif (srf < 1):
    raise Exception('superresolution factor must be one or larger')

  sx = np.array(np.shape(x))
  if sx[0] != 0:
    # check sizes
    if any(sf != sx - sy + 1):
      Exception('size mismatch')

    # initialize PSF
    f = np.linalg.norm(np.ndarray.flatten(y)) / np.linalg.norm(np.ndarray.flatten(x))
    f = f * np.ones(sf) / np.sqrt(np.prod(sf, axis=0))

    # estimate PSF with multiplicative updates
    f = obd_update(f, x, y, maxiter[0], clipping, srf)
    sumf = np.sum(f)
    f = f/sumf # normalize f
    x = sumf*x # adjust x as well
    sx = np.array(np.shape(x))

  else:
    f = np.zeros(sf)
    sf2 = np.ceil(sf/2).astype(np.int32)
    f[sf2[0],sf2[1]] = 1
    sumf = np.sum(f)
    f = f/sumf
    sx = sy + sf - 1
    x = setZero(cnv2tp(f, y, srf));
    return x, f
  # improve true image x with multiplicative updates
  x = obd_update(x, f, y, maxiter[1], clipping, srf);
  return x, f

# function to improve true image x with multiplicative updates
def obd_update(f,x,y,maxiter,clipping,srf): #this is where gradient descent happens in a multiplicative way
#depending on the value of sf the roles of f and x can be swapped
  sf = np.array(f.shape)
  sy = np.array(y.shape)   # for srf > 1, the low resolution y
  m = np.less(y, clipping)*1
  y = np.multiply(y,m)
  for i in range (1, maxiter):
    ytmp = setZero(cnv2(x, f, sy))
    ytmp = np.multiply(ytmp, m)
    num = setZero(cnv2tp(x, y, srf))
    denom = setZero(cnv2tp(x, ytmp, srf))
    tol = 1e-10
    factor = np.divide((num+tol), (denom+tol))
    factor = np.reshape(factor, sf)
    f = np.multiply(f, factor)
  return f #, num, denom, factor, ytmp

# function that converts all negative elements to zero
def setZero(x):
  x[x<0] = 0
  return x

# function that slices an array
def cnv2slice(A, i, j):
  A = A[i,j]
  return A

# function that creates a resizing matrix
# (turns a vector of length n into a vector of length m)
def sampmat(m, n):
    D = np.matmul(np.kron(np.eye(m),np.ones(n)),np.kron(np.eye(n),np.ones(m)).T) / n
    return D

# function that resizes x to have the dimensions sy
def samp2(x, sy):
    sx = x.shape
    y = np.matmul(np.matmul(sampmat(sy[0],sx[0]),x),sampmat(sy[1],sx[1]).T)
    return y

# function that performs 2D convolution via FFTs
def cnv2(x,f,sy):

    # get the dimensions of x and f
    sx = np.array(x.shape)
    sf = np.array(f.shape)

    # if the x image is larger than the PSF, bump up the size of the PSF so that they match
    if np.all(np.greater_equal(sx,sf)):

        # use real FFTs to avoid returning complex values
        y = np.fft.ifft2(np.multiply(np.fft.fft2(x), np.fft.fft2(f, s=[sx[0], sx[1]])))
        # slice out the appropriate piece of the x image to make the y image
        y = cnv2slice(y, slice(sf[0]-1, sx[0]), slice(sf[1]-1, sx[1]))
    # if the PSF is larger than the x image, swap their roles
    elif np.all(np.greater_equal(sf,sx)):
        y = cnv2(f,x,sy)

    # if neither is larger than the other, math has broken and you should alert the world
    else:
        raise Exception('[cnv2] x must be at least as large as f or vice versa.')

    # check that the returned y image matches the expected dimensions
    if np.any(np.greater(sy,y.shape)):
        raise Exception('[cnv2] size missmatch between input and computed y.')

    # if the expected y image dimensions are smaller than the computed one's, downsample it
    if np.any(np.less(sy,y.shape)):
        y = samp2(y,sy)

    # return the computed y image
    return y

def cnv2tp(x, y, srf):
    sx = np.array(x.shape)
    sy = np.array(y.shape)
    if (srf > 1):
        samp2(y, np.floor(srf*sy))
    if np.all(np.greater_equal(sx, sy)):
        sf = sx - sy
        #f = np.fft.ifft2(np.multiply(np.fft.fft2(cnv2slice(x, slice(int(sx[0]/2-sy[0]/2), int(sx[0]/2+sy[0]/2)), slice(int(sx[1]/2-sy[1]/2), int(sx[1]/2+sy[1]/2)))), np.fft.fft2(y)))
        #print(x.shape)
        #print(y.shape)
        #print((np.fft.fft2(cnv2pad(y, sf))).shape)
        #print((np.fft.fft2(x)).shape)
        #fig, ax = plt.subplots(1,2, figsize=(24., 8.))
        #ax[0].imshow(x, origin='lower')
        #plt.show()
        fft_x=np.fft.fft2(x)
        breakpoint()
        pad_y=cnv2pad(y, sf)
        breakpoint()
        fft_y=np.fft.fft2(pad_y)
        breakpoint()
        mult1=np.multiply(fft_x, fft_y)
        breakpoint()
        f=np.fft.ifft2(mult1)
        #f = np.fft.ifft2(np.multiply(np.fft.fft2(x), np.fft.fft2(cnv2pad(y, sf))))
        #pdb.set_trace()
        #f = cnv2slice(np.real(f), slice(int(sy[0]/2-sf[0]/2), int(sy[0]/2+sf[0]/2)), slice(int(sy[1]/2-sf[1]/2), int(sy[1]/2+sf[1]/2)))
        f = cnv2slice(np.real(f), slice(0, sf[0]), slice(0, sf[1]))
        #pdb.set_trace()
    elif np.all(np.less_equal(sx, sy)):
        sf = sy + sx
        f = np.multiply(np.conj(np.fft.fft2(x,s=sf)), np.fft.fft2(cnv2pad(y, sx),s=sf))
        f = np.fft.ifft2(f)
    else:
        raise Exception('[cnv2.m] x must be at least as large as y or vice versa.')
    f = np.real(f)
    return f

def cnv2pad(A, sf):
    i = sf[0]; j = sf[1]
    sA = A.shape
    B = np.zeros((sA[0]+i, sA[1]+j))
    B[i:, j:] = np.real(A)
    return B
