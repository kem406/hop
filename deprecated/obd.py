def obd(x, y, sf, maxiter, clipping, srf):
  
  # if clipping and var don't exist, or clipping is empty, set srf to infinity
  if ~('clipping' in locals() & 'var' in globals()) | clipping == NULL:
    srf = math.inf
  
  # if srf and var don't exist, or srf is empty, set srf to 1
  if ~('srf' in locals() & 'var' in globals()) | srf==NULL:
    srf = 1
  
  # find dimensions of y
  sy = len(y)
  
  # multiply sy and sf by srf and round down to nearest integer if srf above 1
  if (srf > 1):
    sy = math.floor(srf * sy)
    sf = math.floor(srf * sf)

  # srf below 1 is invalid by def. since it is ratio of higher freq to lower freq
  elif (srf < 1):
    raise Exception('superresolution factor must be one or larger')

  if ~(x == NULL):
    # check sizes 
    sx = len(x)
    if any(sf != sx - sy + 1):
      Exception('size mismatch')
    
    # initialize PSF
    f = np.linalg.norm(np.ndarray.flatten(y)) / np.linalg.norm(np.ndarray.flatten(x))
    f = f * np.ones(sf) / np.sqrt(np.prod(sf, axis=0))

    # estimate PSF with multiplicative updates
    f = obd_update(f, x, y, maxiter[0], clipping, srf)
    sumf = sum(math.floor(f))
    f = f/sumf # normalize f
    x = sumf*x # adjust x as well
    sx = len(x);

  else: 
    f = np.zeros(sf)
    sf2 = math.ceil(sf/2)
    f[sf2[0]] <- 1
    f[sf2[1]] <- 1
    f = f/sumf
    sx = sy + sf - 1
    x = setZero(cnv2tp(f, y, srf));
  
  # improve true image x with multiplicative updates
  x = obd_update(x, f, y, maxiter[1], clipping, srf);
  return x, f
