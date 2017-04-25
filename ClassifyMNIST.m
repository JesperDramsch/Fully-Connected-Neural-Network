function out = ClassifyMNIST(data)


names={'params___99-43.mat'; 'params-98-79.mat'; 'params.mat';'params4.mat'};
datacat=[];
for datasets=1:size(names,1)
    load(names{datasets});
    sizedata = max(size(data));
    out = forward_NN([double(data) ones(sizedata,1)],w,n_layers,acttype,false,dropout,dropout_val);
    datacat=[datacat out{end}];
end
data=datacat;

load('params_meta.mat');
sizedata = max(size(data));
out = forward_NN([double(data) ones(sizedata,1)],w,n_layers,acttype,false,dropout,dropout_val);
out=out{end};
% load('params___99-43.mat')
% 
% sizedata = max(size(data));
% out = forward_NN([double(data) ones(sizedata,1)],w,n_layers,acttype,false,dropout,dropout_val);
% out=out{end};

%out=softmax(out')';