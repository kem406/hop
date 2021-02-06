import h5py
import numpy as np
import main.obd.obd as ob
import imageio
import matplotlib.pyplot as plt

filename = 'main/obd/f.hdf5'
with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])

h5py.File.close()

A=np.array(data)
fig, ax = plt.subplots(1,2, figsize=(24., 8.))
ax[0].imshow(A, origin='lower')
plt.show()

tables.file._open_files.close_all()

filename2 = 'main/obd/f2.hdf5'
with h5py.File(filename2, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
