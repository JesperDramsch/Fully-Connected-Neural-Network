close all
clear all
clc;
format compact

train = load('MNIST.mat');
data = double(train.data);
labels = double(train.label);

iterations = 10;
n_layers = 6; %number of layers
neurons = 10; %+bias
m = min(size(data)); %number of inputs
y = min(size(labels)); %number of outputs
eta = 1e-3; % Learning Rate
acttype = 'relu' % sigmoid, tanh or relu
mode = 'mini-batch' % batch, mini-batch or stochastic
dropout = true; % Dropout flag
adaptive = true; % Adaptive learning flag
adaptive_mod=.2; % Adaptive end ratio


%% Condition MNIST
[~,max_i_mnist] = max(labels,[],2);


%% Epoch

sizedata = max(size(data));

best_cost = inf(1);

w=weights_NN(m,y,neurons,n_layers);
killer = true;
checkind = randi([1 60000],2000,1);

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
        warning(sprintf('%s is not a supported batch mode, switching to full batch.',mode))
end 

eta_res=0;
counter=0;
eta_orig=eta;
for epoch = 1:iterations
    i_data = randperm(sizedata);
    for batch=1:floor(sizedata/batchsize);
        points=(batch-1)*batchsize+1:(batch)*batchsize;
            
%% Forward feed
        if dropout
            z = forward_NN([data(i_data(points),:).*(round(1-rand(size(data(i_data(points),:))).^2)) ones(batchsize,1)],w,n_layers,acttype);
        else
            z = forward_NN([data(i_data(points),:) ones(batchsize,1)],w,n_layers,acttype);
        end

%% Backward propagate
        deriv_E_w = backward_NN(z,labels(i_data(points),:),w,n_layers,acttype);
        
%% Evaluate Derivatives 
        for layer=1:n_layers-1
            w{layer} = w{layer}-eta*deriv_E_w{layer};
        end
    end
    
%% Validation
    randval = randi([1 60000],5000,1);
    validate = double(train.data(randval,:));
    validate_label = train.label(randval,:);
    z = forward_NN([validate ones(length(validate(:,1)),1)],w,n_layers,acttype);
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

            eta= eta_orig(1);
            eta_res=epoch;
        end
        if counter > iterations*.35
            break
        end
    end
%% Update Learning rate
    if adaptive
    	%eta = eta * (iterations-epoch)/(iterations)
        eta = (1-adaptive_mod) * eta_orig(1) * cos(2*.5*(epoch-eta_res)*pi/iterations)^2+ adaptive_mod* eta_orig(1);
        eta_orig=[eta_orig;eta];
    end
end
w=backup_w;

%% Wrap it up
indexes = randi([1 6000],1,sizedata);
data = double(train.data(indexes,:));
labels = double(train.label(indexes,:));

%% Test
out = forward_NN([data ones(sizedata,1)],w,n_layers,acttype);
out=out{end};

save('params.mat','w','n_layers','acttype')

%% Plots
%figure
%plotNN(m,neurons,y,n_layers)
%if adaptive
%    figure
%    plot(eta_orig)
%end