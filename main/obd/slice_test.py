import h5py
import numpy as np
import main.obd.obd as ob
import imageio
import matplotlib.pyplot as plt

filename = 'main/obd/f1.hdf5'
with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])

A=np.array(data)
fig, ax = plt.subplots(1,1, figsize=(24., 8.))
ax.imshow(np.abs(A), origin='lower')
plt.show()

filename2 = 'main/obd/f2.h5'
with h5py.File(filename2, "r") as f2:
    # List all groups
    print("Keys: %s" % f2.keys())
    a_group_key = list(f2.keys())[0]

    # Get the data
    data2 = list(f2[a_group_key])

B=np.transpose(np.array(data2))
fig, ax = plt.subplots(1,1, figsize=(24., 8.))
ax.imshow(B, origin='lower')
plt.show()

A=A[0:161,0:181]
A.size
sA = np.array(np.shape(A))
sA
sB = np.array(np.shape(B))
sB
C=np.roll(A,200,0)
C=np.roll(C,200,1)
fig, ax = plt.subplots(1,1, figsize=(24., 8.))
ax.imshow(np.abs(C), origin='lower')
plt.show()
sA
maxSize=99999
for i in range (0,sA[1]):
    for j in range (0,sA[0]):
        C=np.roll(A,i,0)
        C=np.roll(C,j,1)
        D=np.multiply(C,B) - np.multiply(B,B)
        #print (abs(np.sum(D)), i, j)
        print(abs(np.sum(D)))
        print(maxSize)
        if (abs(np.sum(D))<maxSize):
            print(maxSize)
            maxSize=abs(np.sum(D))
maxSize
