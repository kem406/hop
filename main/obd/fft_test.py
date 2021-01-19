import numpy as np
import h5py
import matplotlib.pyplot as plt
#f = np.fft.ifft2(np.multiply(np.fft.fft2(x), np.fft.fft2(cnv2pad(y, sf))))

# f = inverse fast fourier transform of (multiply fft of x and fft of (cnv2pad of y and sf))

def makeGaussian(size, fwhm = 3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

#A = np.random.normal(0, 1, (50, 50))
A = makeGaussian(50, 7, center=(25,25))
fig, ax = plt.subplots(1,3, figsize=(24., 8.))
ax[0].imshow(A, origin='lower')
plt.show()
hf = h5py.File('data.h5', 'w')
hf.create_dataset('gaussian1', data=A)
hf.close
asdasd
