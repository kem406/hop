file = hdf5info('gaussian.hdf5')
A = hdf5read(file.GroupHierarchy.Datasets(1))
A = A*255
image(A)

sf = [50 50]
x = imread('00000001.png');
x = cnv2pad(x, sf)


function B = cnv2pad(A, sf);
% PAD with zeros from the top-left
i = sf(1);  j = sf(2);
[rA, cA] = size(A);
B = zeros(rA+i-1, cA+j-1);
B(i:end, j:end) = A;
end
