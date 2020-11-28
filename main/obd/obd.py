import numpy as np

# function to improve true image x with multiplicative updates 
def obd_update(f,x,y,maxiter,clipping,srf): #this is where gradient descent happens in a multiplicative way
#depending on the value of sf the roles of f and x can be swapped
  sf = len(f)
  sy = len(y)    # for srf > 1, the low resolution y
  m = np.less(y, clipping)*1
  y = np.multiply(y,m)
  for i in range (1, maxiter):
    ytmp = setZero(cnv2(x, f, sy))
    ytmp = np.multiply(ytmp, m)
    num = setZero(cnv2tp(x, y, srf))
    denom = setZero(cnv2tp(x, ytmp, srf))
    tol = 1e-10
    factor = mp.multiply((num+tol), (denom+tol))
    factor = np.reshape(factor, sf)
    f = np.multiply(f, factor)
  return f

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
    sx = x.shape
    sf = f.shape

    # if the x image is larger than the PSF, bump up the size of the PSF so that they match
    if np.all(np.greater_equal(sx,sf)):

        # use real FFTs to avoid returning complex values
        y = np.fft.irfft2(np.multiply(np.fft.rfft2(x), np.fft.rfft2(f, s=(sx[0], sx[1]))))

        # slice out the appropriate piece of the x image to make the y image
        y = cnv2slice(y, slice(sf[0], sx[0]), slice(sf[1], sx[1]))

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
    if np.all(np.greater_equal(sx, sf)):
        sf = sx - sy + 1
        f = np.fft.irfft2(np.multiply(np.fft.rfft2(x), np.fft.rfft2(cnv2pad(y, sf))))
        f = cnv2slice(f, slice(1, sf[0]), slice(1, sf[1]))
    elif np.all(np.less_equal(sx, sf)):
        sf = sy + sx - 1
        f = np.multiply(np.fft.rfft2(x, sf[0], sf[1]), np.fft.rfft2(cnv2pad(y, sx)))
        f = np.fft.irfft2(f, sf[0], sf[1])
    else:
        raise Exception('[cnv2.m] x must be at least as large as y or vice versa.')
    f = np.real(f)
    return f

def cnv2pad(A, sf):
    i = sf[0]; j = sf[1]
    sA = A.shape
    B = np.zeros(sA[0]+i-1, sA[1]+j-1)
    B[i:, j:] = A
    return B
