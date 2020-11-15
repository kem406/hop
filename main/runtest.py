import numpy as np
import matplotlib.pyplot as plt
import ehtim as eh
import obd.obd as ob
from inspect import getmembers, isfunction

epsilon = 1.0e-3

# get a list of the functions in obd
functions_list = np.array([o for o in getmembers(ob) if isfunction(o[1])])

print('---------------------------------------------')

################################################
# test setZero

# This test seems tautological, but fine.

x_inp = np.random.normal(0.0,100.0,1000)
x_out_test = ob.setZero(x_inp)

x_out_true = np.copy(x_inp)
x_out_true[x_out_true < 0.0] *= 0.0

testval = np.sum(x_out_true - x_out_test)

if (testval == 0.0):
    print('function obd.setZero checks out')
else:
    print('!!!!! function obd.setZero is BROKEN !!!!!')

functions_list = np.delete(functions_list,functions_list[:,0] == 'setZero',axis=0)

################################################
# test cnv2slice

# This test seems tautological, but fine.

A_inp = np.random.normal(0.0,100.0,(1000,1000))
i_inp = slice(50,250)
j_inp = slice(500,750)

A_out_test = ob.cnv2slice(A_inp,i_inp,j_inp)

A_out_true = A_inp[i_inp,j_inp]

testval = np.sum(A_out_true - A_out_test)

if (testval == 0.0):
    print('function obd.cnv2slice checks out')
else:
    print('!!!!! function obd.cnv2slice is BROKEN !!!!!')

functions_list = np.delete(functions_list,functions_list[:,0] == 'cnv2slice',axis=0)

################################################
# test sampmat

# This test could be made more sophisticated.
# Currently it just checks the matrix dimensions and
# a couple of trivial limiting cases.

m_inp = 15
n_inp = 25

D_test1 = ob.sampmat(m_inp,m_inp)
D_true1 = np.eye(m_inp)
testval1 = np.sum(D_true1 - D_test1)

D_test2 = ob.sampmat(n_inp,n_inp)
D_true2 = np.eye(n_inp)
testval2 = np.sum(D_true2 - D_test2)

D_test3 = ob.sampmat(m_inp,n_inp)
testval3 = np.sum(np.array(D_test3.shape) - np.array([m_inp,n_inp]))

D_test4 = ob.sampmat(n_inp,m_inp)
testval4 = np.sum(np.array(D_test4.shape) - np.array([n_inp,m_inp]))

testval = testval1 + testval2 + testval3 + testval4

if (testval == 0.0):
    print('function obd.sampmat checks out')
else:
    print('!!!!! function obd.sampmat is BROKEN !!!!!')

functions_list = np.delete(functions_list,functions_list[:,0] == 'sampmat',axis=0)

################################################
# test samp2

# This test could be made more sophisticated.
# Currently it just checks the matrix dimensions.

im_x = eh.image.make_empty(npix=128,fov=100.0*eh.RADPERUAS,ra=12.0,dec=12.0)
im_x = im_x.add_gauss(1.0,beamparams=[30.0*eh.RADPERUAS,30.0*eh.RADPERUAS,0.0,25.0*eh.RADPERUAS,25.0*eh.RADPERUAS])
im_x = im_x.add_gauss(1.0,beamparams=[30.0*eh.RADPERUAS,30.0*eh.RADPERUAS,0.0,-25.0*eh.RADPERUAS,-25.0*eh.RADPERUAS])
x_inp = im_x.imvec.reshape((128,128))

sy_inp = (32,32)

out_test = ob.samp2(x_inp,sy_inp)

im_out_true = eh.image.make_empty(npix=32,fov=100.0*eh.RADPERUAS,ra=12.0,dec=12.0)
im_out_true = im_out_true.add_gauss(1.0,beamparams=[30.0*eh.RADPERUAS,30.0*eh.RADPERUAS,0.0,25.0*eh.RADPERUAS,25.0*eh.RADPERUAS])
im_out_true = im_out_true.add_gauss(1.0,beamparams=[30.0*eh.RADPERUAS,30.0*eh.RADPERUAS,0.0,-25.0*eh.RADPERUAS,-25.0*eh.RADPERUAS])
out_true = im_out_true.imvec.reshape((32,32))

testval = np.sum(np.array(out_test.shape) - np.array(out_true.shape))

if (testval == 0.0):
    print('function obd.samp2 checks out')
else:
    print('!!!!! function obd.samp2 is BROKEN !!!!!')

functions_list = np.delete(functions_list,functions_list[:,0] == 'samp2',axis=0)

################################################
# test cnv2

# This test checks the cnv2 convlution against that
# built into the eht-imaging library.  These are not
# identical constructions, and so small deviations
# are possible; thus the epsilon parameter.

im_x = eh.image.make_empty(npix=128,fov=100.0*eh.RADPERUAS,ra=12.0,dec=12.0)
im_x = im_x.add_gauss(1.0,beamparams=[3.0*eh.RADPERUAS,3.0*eh.RADPERUAS,0.0,25.0*eh.RADPERUAS,25.0*eh.RADPERUAS])
im_x = im_x.add_gauss(1.0,beamparams=[3.0*eh.RADPERUAS,3.0*eh.RADPERUAS,0.0,-25.0*eh.RADPERUAS,-25.0*eh.RADPERUAS])
x_inp = im_x.imvec.reshape((128,128))

im_f = eh.image.make_empty(npix=32,fov=25.0*eh.RADPERUAS,ra=12.0,dec=12.0)
im_f = im_f.add_gauss(1.0,beamparams=[6.0*eh.RADPERUAS,3.0*eh.RADPERUAS,0.0,0.0,0.0])
f_inp = im_f.imvec.reshape((32,32))
sy_inp = (96,96)

y_test = ob.cnv2(x_inp,f_inp,sy_inp)
im_y = eh.image.make_empty(npix=len(y_test),fov=75.0*eh.RADPERUAS,ra=12.0,dec=12.0)
im_y.imvec = y_test.ravel()

im_conv = im_x.blur_gauss(beamparams=[6.0*eh.RADPERUAS,3.0*eh.RADPERUAS,0.0,0.0,0.0])
y_true_temp = im_conv.imvec.reshape(128,128)
y_true = np.copy(y_true_temp[17:113,17:113])
im_y_true = im_y.copy()
im_y_true.imvec = y_true.ravel()

normdiff = (y_true - y_test)/np.mean(y_true)
testval = np.max(np.abs(normdiff))

if (testval < epsilon):
    print('function obd.cnv2 checks out')
else:
    print('!!!!! function obd.cnv2 is BROKEN !!!!!')

functions_list = np.delete(functions_list,functions_list[:,0] == 'cnv2',axis=0)

################################################

print('---------------------------------------------')
print('The following functions have not been tested:')
for f in functions_list:
    print('obd.'+f[0])


