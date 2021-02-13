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
maxSize=abs(np.sum(np.multiply(A,B) - np.multiply(B,B)))
maxSize
graph=np.ones((181,161))
graph.shape
maxSize=abs(np.sum(np.multiply(A,B) - np.multiply(B,B)))
for i in range (0,sA[1]):
    for j in range (0,sA[0]):
        C=np.roll(A,i,1)
        C=np.roll(C,j,0)
        D=np.multiply(C,B) - np.multiply(B,B)
        graph[i,j]=abs(np.sum(D))
        if (np.greater(abs(np.sum(D)),maxSize)):
            maxSize=abs(np.sum(D))
            #fig, ax = plt.subplots(1,1, figsize=(24., 8.))
            #ax.imshow(np.abs(C), origin='lower')
            #plt.show()
            print(maxSize,i,j)
maxSize
fig, ax = plt.subplots(1,1, figsize=(24., 8.))
ax.imshow(np.abs(graph), origin='lower')
plt.show()
E=np.roll(A,139,0)
E=np.roll(E,171,1)
fig, ax = plt.subplots(1,1, figsize=(24., 8.))
ax.imshow(np.abs(E), origin='lower')
plt.show()
fig, ax = plt.subplots(1,1, figsize=(24., 8.))
ax.imshow(np.abs(B), origin='lower')
plt.show()

abs(np.sum(np.multiply(B,B) - np.multiply(B,B)))
abs(np.sum(A))
abs(np.sum(B))
