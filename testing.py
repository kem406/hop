##testing region

imagepath = 'obd/data/epsilon_lyrae/'

def y_fname(i):
    return imagepath+"{0:08d}".format(i)+'.png'

test = imageio.imread(y_fname(10))

plt.imshow(test)

np.zeros(np.array([64,64]))

x, f = obd([],z,np.array([64, 64]),[50,1])
plt.imshow(x)

x, f = obd(x,z,np.array([64, 64]),[50,1])
plt.imshow(x)

plt.imshow(f)

num = obd(x,z,np.array([64, 64]),[50,2])

plt.imshow(num[:64,:64])

plt.imshow(num)

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
