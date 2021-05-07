import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def read_fits(filename):
    with fits.open(filename) as hdul:
        data = hdul[0].data
        info = hdul[0].header
        data = data.astype(np.float64)
    return data

def debayer(inpname):

    data = read_fits(inpname)

    # blue channel
    chan_blue = data[1::2,1::2]

    # red channel
    chan_red = data[::2,::2]

    # green channel
    chan_green_temp = np.zeros((int(data.shape[0]/2),data.shape[1]))
    for i in range(data.shape[1]):
        if ((i % 2) == 0):
            chan_green_temp[:,i] = data[1::2,i]
        else:
            chan_green_temp[:,i] = data[::2,i]
    chan_green = 0.5*(chan_green_temp[:,::2] + chan_green_temp[:,1::2])

    # construct image array
    im = np.zeros((chan_blue.shape[0],chan_blue.shape[1],3))
    im[:,:,0] = chan_red
    im[:,:,1] = chan_green
    im[:,:,2] = chan_blue

    return im

def fits_to_png(inpname,outname,maxval=65535.0):
    im = debayer(inpname)
    im /= maxval
    plt.imsave(outname,im)
