close all
clear all
clc;
format compact

train = load('MNIST_train.mat');
labels = double(train.label);
data = double(train.data);
%indexes = randi([1 6000],1,5000);
%data = double(train.data(indexes,:));
%labels = double(train.label(indexes,:));



iterations = 1000;
n_layers = 3; %number of layers 
neurons = 800; %+bias
m = min(size(data)); %number of inputs
y = min(size(labels)); %number of outputs
eta = 1e-6; % Learning Rate
acttype = 'relu' % sigmoid, tanh or relu
mode = 'mini-batch' % batch, mini-batch or stochastic
training = true;
dropout = true; % Dropout flag
dropout_val = [.2 .5];
outname='params.mat';

%% Condition MNIST
[~,max_i_mnist] = max(labels,[],2);

%% Epoch

sizedata = max(size(data))-10000;
best_cost = inf(1);
w=weights_NN(m,y,neurons,n_layers);
delta_w=cellfun(@(x) x*0,w,'un',0);

switch mode
    case 'mini-batch'
        batchsize = 50;
    case 'batch'
    	batchsize = sizedata;
        if (dropout && iterations<=5)
            warning('Batch-mode, setting dropout to false.')
            dropout = false;
        end
	case 'stochastic'
    	batchsize = 1;
    otherwise
        batchsize = sizedata;
        warning('%s is not a supported batch mode, switching to full batch.',mode)
end 
backup_w={};
counter=0;
for epoch = 1:iterations
    i_data = randperm(sizedata+10000);
    i_test = i_data(1:sizedata);
    for batch=1:floor(sizedata/batchsize)
        points=(batch-1)*batchsize+1:(batch)*batchsize;
            
%% Forward feed
        z = forward_NN([data(i_test(points),:) ones(batchsize,1)],w,n_layers,acttype,training,dropout,dropout_val);


%% Backward propagate
        deriv_E_w = backward_NN(z,labels(i_test(points),:),w,n_layers,acttype);
        
%% Evaluate Derivatives
        new_w = update_NN(w,delta_w,deriv_E_w,n_layers,eta);
        delta_w = cellfun(@minus,w,new_w,'Un',0);
        w=new_w;
    end
    
%% Validation
    %randval=indexes;
    i_validate = i_data(sizedata+1:end);
    validate = double(data(i_validate,:));
    validate_label = train.label(i_validate,:);
    z = forward_NN([validate ones(length(validate(:,1)),1)],w,n_layers,acttype,false,dropout,dropout_val);
    cost= costfunction(z{end},validate_label,'RMS');
    %totalerror=mnist_error(max_i_mnist(:),z{end});
    totalerror=mnist_error(max_i_mnist(i_validate),z{end});
    fprintf('Iteration: %i, Cost: %f; Error: %f%%\n',epoch, cost, totalerror)
    if cost < best_cost
        if epoch>50 && totalerror < 5
        fprintf('Saving\n') 
        parsave(outname,w,n_layers,acttype,dropout,dropout_val)
        fprintf('Done\n')
        end
        best_cost = cost;
        counter = 0;
        backup_w = w;
        
    else
        counter = counter +1;
        if counter > iterations*.35
            break
        end
    end

end
w=backup_w;

parsave(outname,w,n_layers,acttype,dropout,dropout_val)
fprintf('Finished.')
%% Plots
%figure
%plotNN(m,neurons,y,n_layers)
