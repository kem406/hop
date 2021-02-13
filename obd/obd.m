function [x, f] = obd(x, y, sf, maxiter, clipping, srf);
%OBD does a single step for Online Blind Deconvolution.
%
% Purpose:
%   Implements the algorithm in:
%
%      Stefan Harmeling, Michael Hirsch, Suvrit Sra, Bernhard Sch\"olkopf,
%      "Online Blind Image Deconvolution for Astronomy", in Proceedings of
%      the First International Conference on Computational Photography, 2009.
%
% Inputs:
%   x              - current estimate of the deblurred image
%                    possible initialization is ones(sx), default == [],
%                    if x is empty, then x will be created by using y with 
%                    the appropriate sizes depending on 
%   y              - next observed image
%   sf             - size of the PSF (in the scale of y, i.e. independent of srf)
%   maxiter        - e.g. [100, 3] means 100 update steps for f and 3 for x
%   clipping       - pixel value of saturated pixels (e.g. 656535), default == Inf
%   srf            - super-resolution factor
%
% Note that the size of x determines the size of the PSF, via
%         size(f) = size(x) - size(y) + 1
%
% Outputs:
%   x              - new estimate of the deblurred image
%
% Copyright (C) 2010 by Michael Hirsch & Stefan Harmeling.

if ~exist('clipping', 'var') || isempty(clipping), srf = Inf; end
if ~exist('srf', 'var') || isempty(srf), srf = 1; end

sy = size(y);            % size of blurred image   
if srf > 1
  sy = floor(srf * sy);
  sf = floor(srf * sf);
elseif srf < 1
  error('superresolution factor must be one or larger');
end

if ~isempty(x)
  % check sizes
  sx = size(x);
  if any(sf ~= sx - sy + 1)
    error('size missmatch')
  end

  % intialize PSF
  f = norm(y(:)) / norm(x(:));
  f = f * ones(sf) / sqrt(prod(sf));

  % estimate PSF with multiplicative updates
  f = obd_update(f, x, y, maxiter(1), clipping, srf);
  sumf = sum(f(:));
  fprintf("sumf:");
  fprintf('%f\n', sum(f(:)));
  fprintf("sumx before:");
  fprintf('%f\n', sum(x(:)));
  f = f/sumf;                         % normalize f
  x = sumf*x;                         % adjust x as well
  fprintf("sumx after:");
  fprintf('%f\n', sum(x(:)));
 sx = size(x);
 
else
  f = zeros(sf);
  sf2 = ceil(sf/2);
  f(sf2(1), sf2(2)) = 1;   % a delta peak
  f = f/sum(f(:));
  sx = sy + sf - 1;
  x = pos(cnv2tp(f, y, srf));
  return
end

% improve true image x with multiplicative updates
x = obd_update(x, f, y, maxiter(2), clipping, srf);
return

%%%%%%%%%%%%%%%%
function f = obd_update(f, x, y, maxiter, clipping, srf)
% depending on the value of sf the roles of f and x can be swapped
sf = size(f);
sy = size(y);    % for srf > 1, the low resolution y
m = (y < clipping);     % where do we have clipping
y = y .* m;                      % deal with clipping
for i = 1:maxiter
  ytmp = pos(cnv2(x, f, sy));
  ytmp = ytmp .* m;                 % deal with clipping
  nom = pos(cnv2tp(x, y, srf));
  denom = pos(cnv2tp(x, ytmp, srf));
  tol = 1e-10;
  factor = (nom+tol) ./ (denom+tol);
  factor = reshape(factor, sf);
  f = f .* factor;
end
return

%%%%%%%%%%%%%%%%%
function x = pos(x, epsilon)
x(find(x(:)<0)) = 0;
return

%%%%%%%%%%%%%%%%%
function A = cnv2slice(A, i, j);
A = A(i,j);
return

%%%%%%%%%%%%%%%%%
function y = cnv2(x, f, sy)
sx = size(x);
sf = size(f);
if all(sx >= sf)   % x is larger or equal to f
  % perform convolution in Fourier space
  y = ifft2(fft2(x) .* fft2(f, sx(1), sx(2)));
  y = cnv2slice(y, sf(1):sx(1), sf(2):sx(2));
elseif all(sx <= sf)  % x is smaller or equal than f
  y = cnv2(f, x, sy);
else
  % x and f are incomparable
  error('[cnv2.m] x must be at least as large as f or vice versa.');
end
if any(sy > size(y))
  error('[cnv2.m] size missmatch');
end
if any(sy < size(y));
  y = samp2(y, sy);   % downsample
end
return

%%%%%%%%%%%%%%%%%
function f = cnv2tp(x, y, srf)
sx = size(x);
if srf > 1
  y = samp2(y, floor(srf*size(y)));    % upsample
end
sy = size(y);
  
% perform the linear convolution in Fourier space
if all(sx >= sy)
  sf = sx - sy + 1;
  %placeholder=0;
  fft_x=conj(fft2(x));
  %clf
  %subplot(131), imagesc(abs(fftshift(fft_x))), title(sprintf('fft_x')); axis equal, axis tight
  %drawnow
  pad_y=cnv2pad(y, sf);
  %clf
  %subplot(131), imagesc(pad_y), title(sprintf('pad_y')); axis equal, axis tight
  %drawnow  
  fft_y=fft2(pad_y);
  %clf
  %subplot(131), imagesc(abs(fftshift(fft_y))), title(sprintf('fft_y')); axis equal, axis tight
  %drawnow  
  mult1=fft_x.*fft_y;
  %clf
  %subplot(131), imagesc(abs(fftshift(mult1))), title(sprintf('mult1')); axis equal, axis tight
  %drawnow
  ifft_xy=ifft2(mult1);
  if ~isfile('f2.h5') || h5info('f2.h5').Name ~= '/'
    h5create('f2.h5','/myDataset',size(ifft_xy));
  end
  h5write('f2.h5','/myDataset',ifft_xy);
  %clf
  %subplot(131), imagesc(ifft_xy), title(sprintf('ifft_xy')); axis equal, axis tight
  %drawnow
  f=cnv2slice(ifft_xy, 1:sf(1), 1:sf(2));
  %clf
  %subplot(131), imagesc(f), title(sprintf('f')); axis equal, axis tight
  %drawnow
  %placeholder=1;
  
  %f = cnv2slice(ifft2(conj(fft2(x)).*fft2(cnv2pad(y, sf))), 1:sf(1), 1:sf(2));
elseif all(sx <= sy)
  sf = sy + sx - 1;
  f = ifft2(conj(fft2(x, sf(1), sf(2))).*fft2(cnv2pad(y, sx), sf(1), sf(2)));
else  % x and y are incomparable
  error('[cnv2.m] x must be at least as large as y or vice versa.');
end
f = real(f);  
return

%%%%%%%%%%%%%
function B = cnv2pad(A, sf);
% PAD with zeros from the top-left
i = sf(1);  j = sf(2);
[rA, cA] = size(A);
B = zeros(rA+i-1, cA+j-1);
B(i:end, j:end) = A;
return

%%%%%%%%%%%%%
function y = samp2(x, sy);
sx = size(x);
% downsample by factor srf
y = sampmat(sy(1), sx(1)) * x * sampmat(sy(2), sx(2))';
return

%%%%%%%%%%%%%
function D = sampmat(m, n)
D = kron(speye(m), ones(n, 1)') * kron(speye(n), ones(m, 1))/m;
return
%%%%% END OF obd.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
