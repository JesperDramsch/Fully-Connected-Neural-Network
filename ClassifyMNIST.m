function out = ClassifyMNIST(data)

load('params.mat')
sizedata = max(size(data));
out = forward_NN([double(data) ones(sizedata,1)],w,n_layers,acttype,false,dropout,dropout_val);
out=out{end};