close all
clear all
clc;
format compact

train = load('MNIST.mat');
data = double(train.data);
labels = double(train.label);

iterations = 1000;
n_layers = 4; %number of layers
neurons = 200; %+bias
m = min(size(data)); %number of inputs
y = min(size(labels)); %number of outputs
eta = 1e-5; % Learning Rate
acttype = 'relu' % sigmoid, tanh or relu
mode = 'mini-batch' % batch, mini-batch or stochastic
training = true;
dropout = true; % Dropout flag
dropout_in = .8;
dropout_hidden = .5;

%% Condition MNIST
[~,max_i_mnist] = max(labels,[],2);


%% Epoch

sizedata = max(size(data));
best_cost = inf(1);
w=weights_NN(m,y,neurons,n_layers);
delta_w=cellfun(@(x) x*0,w,'un',0);

switch mode
    case 'mini-batch'
        batchsize = 25;
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

counter=0;
for epoch = 1:iterations
    i_data = randperm(sizedata);
    for batch=1:floor(sizedata/batchsize)
        points=(batch-1)*batchsize+1:(batch)*batchsize;
            
%% Forward feed
        z = forward_NN([data(i_data(points),:) ones(batchsize,1)],w,n_layers,acttype,training,dropout,dropout_in,dropout_hidden);


%% Backward propagate
        deriv_E_w = backward_NN(z,labels(i_data(points),:),w,n_layers,acttype);
        
%% Evaluate Derivatives
        new_w = update_NN(w,delta_w,deriv_E_w,n_layers,eta);
        delta_w = cellfun(@minus,w,new_w,'Un',0);
        w=new_w;
    end
    
%% Validation
    randval = randi([1 60000],5000,1);
    validate = double(train.data(randval,:));
    validate_label = train.label(randval,:);
    z = forward_NN([validate ones(length(validate(:,1)),1)],w,n_layers,acttype,false,dropout,dropout_in,dropout_hidden);
    cost= costfunction(z{end},validate_label,'RMS');
    totalerror=mnist_error(max_i_mnist(randval),z{end});
    fprintf('Iteration: %i, Cost: %f; Error: %f%%\n',epoch, cost, totalerror)
    if cost < best_cost
        best_cost = cost;
        counter = 0;
        backup_w = w;
    else
        counter = counter +1;
        if counter == round(iterations*.05)
            w=backup_w;
        end
        if mod(iterations,counter)==50
            w=weights_NN(m,y,neurons,n_layers);

        end
        if counter > iterations*.35
            break
        end
    end

end
w=backup_w;

%% Wrap it up
indexes = randi([1 6000],1,sizedata);
data = double(train.data(indexes,:));
labels = double(train.label(indexes,:));

%% Test
out = forward_NN([data ones(sizedata,1)],w,n_layers,acttype,training,dropout,dropout_in,dropout_hidden);
out=out{end};

save('params.mat','w','n_layers','acttype','dropout','dropout_in','dropout_hidden')

%% Plots
%figure
%plotNN(m,neurons,y,n_layers)
