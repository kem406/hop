def obd_update(f,x,y,maxiter,clipping,srf): #this is where gradient descent happens in a multiplicative way 
#depending on the value of sf the roles of f and x can be swapped
  sf = len(f)
  sy = len(y)    # for srf > 1, the low resolution y  
  m = np.less(y, clipping)*1
  y = np.multiply(y,m)
  for i in range (1, maxiter):
    ytmp = setZero(cnv2(x, f, sy))
    ytmp = np.multiply(ytmp, m)
    num = setZero(cnv2tp(x, y, srf))
    denom = pos(cnv2tp(x, ytmp, srf))
    tol = 1e-10
    factor = mp.multiply((num+tol), (denom+tol))
    factor = np.reshape(factor, sf)
    f = np.multiply(f, factor)
  return f
