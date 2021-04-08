h5create('myfile.h5','/myDataset',[100 200 300])
mydata = ones(100,200,300);
h5write('myfile.h5','/myDataset',mydata)
h5disp('myfile.h5')