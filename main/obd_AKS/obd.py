import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import filters

#Rewrite with everything square and consistent
#All images will be in powers of 2 square size for ease... just massage data to be so

def obd(x,y,sf,maxiter):
    # x is estimate of deblurred image
    # y is observed image
    # sf size of psf
    # max iters for gradient descent

    # we will stay consistent with the notation to keep f as the psf
    sx = np.array(np.shape(x))
    sy = np.array(np.shape(y))

    ### Update/Guess the PSF

    #if there is already a guess for x, use it to guess f
    if sx[0] != 0:
        # initialize PSF as flat w/ correct intensity
        f = np.linalg.norm(np.ndarray.flatten(y)) / np.linalg.norm(np.ndarray.flatten(x))
        f = f * np.ones(sf) / np.sqrt(np.prod(sf, axis=0))

        #lets do GD on f given x and y
        #obd update(f,x,y)
        for i in range(0,maxiter[0]):
            #I am everywhere here making assumptions about sx,sf,and sy.
            #Just let me do that a minute please.
            ytmp = np.multiply(np.fft.fft2(x,s=sx), np.fft.fft2(f, s=sx))
            ytmp = setZero(np.real(np.fft.ifft2(ytmp)))[sf[0]-1:,sf[1]-1:] #so they do not seem to do the np.real here... what does pos mean in that case?

            Y = np.zeros(sx)
            Y[sf[0]-1:,sf[1]-1:] = y
            num = np.multiply(np.conj(np.fft.fft2(x,s=sx)),np.fft.fft2(Y,s=sx))
            num = setZero(np.real(np.fft.ifft2(num)))[:sf[0],:sf[0]]

            Y = np.zeros(sx) #,dtype=np.complex64)
            Y[sf[0]-1:,sf[1]-1:] = ytmp
            denom = np.multiply(np.conj(np.fft.fft2(x,s=sx)),np.fft.fft2(Y,s=sx))
            denom = setZero(np.real(np.fft.ifft2(denom)))[:sf[0],:sf[0]]

            tol = 1e-10
            factor = np.divide((num+tol),(denom+tol))
            factor = factor*filters.window(('tukey',0.3),(sf[0],sf[1]),warp_kwargs={'order':3}) #attempt to eliminate edge spikes
            f = np.multiply(f, factor)

        #this normalization seem suspect for making the light curve
        sumf = np.sum(f)
        f = f/sumf # normalize f
        x = sumf*x # adjust x as well
        #so this is shifting all the power from f to x
        #f is always unit normalized
        #now we guess the structure of x given y and that we have
        #renormalized x to have the same power as y
        ## actually we are normalizing by abs(image) not image
        ## power. This makes me feel uncomfortable.

    #now that we have good guess for f, use it to guess x given y
        #lets do GD on x given f and y
        #obd update(x,f,y)
        for i in range(0,maxiter[1]):
            #I am everywhere here making assumptions about sx,sf,and sy.
            #Just let me do that a minute please.
            ytmp = np.multiply(np.fft.fft2(x,s=sx), np.fft.fft2(f, s=sx))
            ytmp = setZero(np.real(np.fft.ifft2(ytmp)))[sf[0]-1:,sf[1]-1:] #so they do not seem to do the np.real here... what does pos mean in that case?

            Y = np.zeros(sx)
            Y[sf[0]-1:,sf[1]-1:] = y
            num = np.multiply(np.conj(np.fft.fft2(f,s=sx)),np.fft.fft2(Y,s=sx))
            num = setZero(np.real(np.fft.ifft2(num)))

            Y = np.zeros(sx) #,dtype=np.complex64)
            Y[sf[0]-1:,sf[1]-1:] = ytmp
            denom = np.multiply(np.conj(np.fft.fft2(f,s=sx)),np.fft.fft2(Y,s=sx))
            denom = setZero(np.real(np.fft.ifft2(denom)))

            tol = 1e-10
            factor = np.divide((num+tol),(denom+tol))
            factor = factor*filters.window(('tukey',0.3),(sx[0],sx[1]),warp_kwargs={'order':3}) #attempt to eliminate edge spikes
            x = np.multiply(x, factor)

        return x, f

    #intialization of f from scratch
    else:
        f = np.zeros(sf)
        mid = int(f.shape[0]/2)
        f[mid,mid] = 1 #delta function intialization
        # make the guess for x to be size of sy padded by sf

    #using our intialization of f, use it to guess x given y
        ## here I am assuming sf<sy
        sx = sf + sy - 1
        Y = np.zeros(sx)
        Y[sf[0]-1:,sf[1]-1:] = y
        x = np.multiply(np.conj(np.fft.fft2(f,s=sx)),np.fft.fft2(Y,s=sx))
        x = setZero(np.real(np.fft.ifft2(x)))
        ## to be clear, this is a waste of time, beacuse we know we are choosing x=y1
        ## that is what the delta function means.
        ## This was useful coding practice because it means the image is centered using my conventions
        ## need to understand why this padding is really necessary
        ## these lines may be useful for srf cases which we are ignoring rn
        return x, f

# function that converts all negative elements to zero
def setZero(x):
    x[x<0] = 0
    return x
