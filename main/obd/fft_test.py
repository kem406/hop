import h5py
import numpy as np
import main.obd.obd as ob
import imageio
import matplotlib.pyplot as plt

def cnv2pad(A, sf):
    i = sf[0]
    j = sf[1]
    sA = A.shape
    B = np.zeros((sA[0]+i, sA[1]+j))
    B[i:, j:] = np.real(A)
    return B

filename = 'main/obd/gaussian.hdf5'
with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
A=np.array(data)
fig, ax = plt.subplots(1,2, figsize=(24., 8.))
ax[0].imshow(A, origin='lower')
plt.show()

x = np.array(imageio.imread('obd/data/epsilon_lyrae/00000001.png'))
x = x[:,:,1].astype(np.float64)
y = np.array(imageio.imread('obd/data/epsilon_lyrae/00000002.png'))
y = y[:,:,1].astype(np.float64)
sf=np.array([50,50])
x=cnv2pad(x, sf)

#test every part of:
#f = np.fft.ifft2(np.multiply(np.fft.fft2(x), np.fft.fft2(cnv2pad(y, sf))))

fft1=np.fft.fft2(x)
fig, ax = plt.subplots(1,2, figsize=(24., 8.))
ax[0].imshow(abs(np.fft.fftshift(fft1)), origin='lower')
plt.show()

pad1=cnv2pad(y, sf)
fig, ax = plt.subplots(1,2, figsize=(24., 8.))
ax[0].imshow(pad1, origin='lower')
plt.show()

fft2=np.fft.fft2(cnv2pad(y, sf))
fig, ax = plt.subplots(1,2, figsize=(24., 8.))
ax[0].imshow(abs(np.fft.fftshift(fft2)), origin='lower')
plt.show()

mult1=np.multiply(fft1, fft2)
fig, ax = plt.subplots(1,2, figsize=(24., 8.))
ax[0].imshow(abs(mult1), origin='lower')
plt.show()

ifft1 = np.fft.ifft2(mult1)
fig, ax = plt.subplots(1,2, figsize=(24., 8.))
ax[0].imshow(abs(ifft1), origin='lower')
plt.show()
