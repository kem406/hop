import numpy as np

x=[1]
len(x)

type(np.ceil(50).astype(np.int32))

np.sum(np.ones((256,256)))

size_fft = [int(256),int(256)]
size_fft[0]
np.fft.fft2(np.ones((256,256)),s=size_fft)

A = np.ones((112,132))
sf = np.array([50,50])
i = sf[0]; j = sf[1]
sA = A.shape
B = np.zeros((sA[0]+i, sA[1]+j))
B[i:, j:] = A

B[i:, j:].shape
B.shape
A.shape

(50,50)+(112,132)
