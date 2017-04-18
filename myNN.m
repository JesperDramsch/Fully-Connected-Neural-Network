close all;
clear all;
clc;
format compact

addpath('..')
data_type=1;
data_points=1000;
data_noise=.95;
train = getDataNN(data_type,data_points,data_noise,1);

data = train(:,1:2);
labels = train(:,3:4);

iterations = 1000;
n_layers = 5; %number of layers
neurons = 4; %+bias
m = 2; %number of inputs
y = 2; %number of outputs
eta = 1e-3; % Learning Rate
type = 'relu' % sigmoid, tanh or relu
mode = 'mini-batch' % batch, mini-batch or stochastic
dropout = true; % Dropout flag
adaptive = true; % Adaptive learning flag
adaptive_mod=.2; % Adaptive end ratio


sizedata = size(data,1);

best_cost = inf(1);

w=weights_NN(m,y,neurons,n_layers);
killer = true;
while killer
    try
        check_gradients(data,labels,w,n_layers,type)
    catch
        w=weights_NN(m,y,neurons,n_layers);
    	continue
    end
	killer=false;
end
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
            z = forward_NN([data(i_data(points),:).*(round(1-rand(size(data(i_data(points),:))).^2)) ones(batchsize,1)],w,n_layers,type);
        else
            z = forward_NN([data(i_data(points),:) ones(batchsize,1)],w,n_layers,type);
        end

%% Backward propagate
        deriv_E_w = backward_NN(z,labels(i_data(points),:),w,n_layers,type);
        
%% Evaluate Derivatives 
        for layer=1:n_layers-1
            w{layer} = w{layer}-eta*deriv_E_w{layer};
        end
    end
    
%% Validation
    validate = getDataNN(data_type,data_points,data_noise,0);
    z = forward_NN([validate(:,1:2) ones(length(validate(:,1)),1)],w,n_layers,type);
    cost= costfunction(z{end},validate(:,3:4),'RMS');
    totalerror=100-sum(round(z{end}(:,2)) == validate(:,4))/sizedata*100;
    display(sprintf('Iteration: %i, Cost: %f; Error: %f%%',epoch, cost, totalerror))
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
            killer = true;
            w=weights_NN(m,y,neurons,n_layers);
            while killer
                try
                    check_gradients(data,labels,w,n_layers,type)
                catch
                    w=weights_NN(m,y,neurons,n_layers);
                    continue
                end
                killer=false;
                eta= eta_orig(1);
                eta_res=epoch;
            end
                
            
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


data=validate(:,1:2);
labels=validate(:,3:4);

%% Test
out = forward_NN([data ones(sizedata,1)],w,n_layers,type);
out=out{end};
totalerror=100-sum(round(out(:,2)) == labels(:,2))/sizedata*100;
display(sprintf('The total missclassification in the best run is %0.2f%%.',totalerror))


%% Plots

figure
plotNN(m,neurons,y,n_layers)

if adaptive
    figure
    plot(eta_orig)
end

figure
id = find(out(:,1)>.5);
plot(data(id,1), data(id,2), 'b.', 'MarkerSize', 20);
hold on
id = find(out(:,2)>.5);
plot(data(id,1), data(id,2), 'r.', 'MarkerSize', 20);
id = find(round(out(:,1)) ~= labels(:,1));
plot(data(id,1), data(id,2), 'g.', 'MarkerSize', 5);
id = find(round(out(:,2)) ~= labels(:,2));
plot(data(id,1), data(id,2), 'g.', 'MarkerSize', 5);
if totalerror > 0
    legend('Class 1', 'Class 2', 'Missclassified')
else
    legend('Class 1', 'Class 2')
end