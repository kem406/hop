import numpy as np
import matplotlib.pyplot as plt

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
        f= x

    #now that we have good guess for f, use it to guess x given y

        return x, f
    #intialization of f from scratch
    else:
        f = np.zeros(sf)
        mid = int(f.shape[0]/2)
        f[mid,mid] = 1 #delta function intialization
        # make the guess for x to be size of sy padded by sf

    #using our intialization of f, use it to guess x given y
        ## here I am assuming sf<sy
        sx = sf + sy
        Y = np.zeros(sx)
        Y[sf[0]:,sf[1]:] = y
        x = np.multiply(np.conj(np.fft.fft2(f,s=sx)),np.fft.fft2(Y,s=sx))
        x = np.real(np.fft.ifft2(x))
        ## to be clear, this is a waste of time, beacuse we know we are choosing x=y1
        ## that is what the delta function means.
        ## This was useful coding practice because it means the image is centered using my conventions
        ## need to understand why this padding is really necessary
        return x, f

# function that converts all negative elements to zero
def setZero(x):
    x[x<0] = 0
    return x

##testing region

np.zeros(np.array([64,64]))

x,f = obd([],z,np.array([64, 64]),[50,2])

plt.imshow(x[:256,:256])

plt.imshow(x)


np.unravel_index(np.argmax(x), x.shape)


plt.imshow(f)

test = np.zeros((256,256))

def star_flux(flux,sigma_x = 6.,sigma_y = 12.,size = 256):
    x = np.linspace(-size/2, size/2-1, size)
    y = np.linspace(-size/2, size/2-1, size)
    x, y = np.meshgrid(x, y)

    z = flux*(1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
         + y**2/(2*sigma_y**2))))
    return z

z = star_flux(1)

plt.imshow(z)


test.shape[0]

test1 = np.zeros((256,256))
test1[128,128] = 1
np.fft.fftshift(test1)

np.array([50, 50])[0]/2
