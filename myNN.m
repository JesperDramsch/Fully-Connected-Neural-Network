close all;
clear all;
clc;
format compact

addpath('..')
train = getDataNN(2,1000,.2,1);
validate = getDataNN(2,1000,.2,0);

data = train(:,1:2);
labels = train(:,3:4);

iterations = 100;
n_layers = 3; %number of layers
neurons = 4; %+bias
m = 2; %number of inputs
y = 2; %number of outputs
eta = 1e-2;
type = 'relu' % sigmoid, tanh or relu
mode = 'mini-batch' % batch, mini-batch or stochastic
dropout = true; % Dropout flag
adaptive = true; % Adaptive learning flag



sizedata = size(data,1);
unimod = 1*m^-.5;
best_cost = inf(1);
w = cell([n_layers-1, 1]);
w(1) = {-unimod+(unimod+unimod)*(rand([m+1, neurons]))};
for layers = 2:n_layers-2
    w(layers) = {-unimod+(unimod+unimod)*(rand(neurons+1,neurons))};
end
w(n_layers-1) = {-unimod+(unimod+unimod)*(rand([neurons+1,y]))};

check_gradients(data,labels,w,n_layers,type)
counter=0;

figure
plotNN(m,neurons,y,n_layers)
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
    z = forward_NN([validate(:,1:2) ones(length(validate(:,1)),1)],w,n_layers,type);
    cost= costfunction(z{end},validate(:,3:4),'RMS');
    totalerror=100-sum(round(z{end}(:,2)) == validate(:,4))/sizedata*100;
    display(sprintf('Iteration: %i, Cost: %f; Error: %f%%',epoch, cost, totalerror))
    if cost < best_cost
        best_cost = cost;
    else
        counter = counter +1;
        if counter > iterations*.1
            break
        end
    end
%% Update Learning rate
    if adaptive
    	%eta = eta * (iterations-epoch)/(iterations)
        eta = .5 * eta_orig(1) * cos(.5*epoch*pi/iterations)^2+ .5* eta_orig(1);
        eta_orig=[eta_orig;eta];
    end
end



if adaptive
    figure
    plot(eta_orig)
end
%% Test
out = forward_NN([data ones(sizedata,1)],w,n_layers,type);
out=out{end};
totalerror=100-sum(round(out(:,2)) == labels(:,2))/sizedata*100;
display(sprintf('The total missclassification in the final run is %0.2f%%.',totalerror))


%% Plot
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