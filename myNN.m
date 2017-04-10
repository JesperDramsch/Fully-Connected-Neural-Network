close all;
clear all;
clc;
format compact

addpath('..')
data = getDataNN(2,1000,.05,1);

iterations = 100;
n_layers = 5; %number of layers
neurons = 4; %+bias
m = 2; %number of inputs
y = 2; %number of outputs
eta = 1e-2;
type = 'relu' % sigmoid, tanh or relu
mode = 'stochastic' % batch, mini-batch or stochastic
dropout = true; % Dropout flag
adaptive = true; % Adaptive learning flag

figure
plotNN(m,neurons,y,n_layers)

sizedata = size(data,1);
unimod = m^-.5;

w = cell([n_layers-1, 1]);
w(1) = {-unimod+2*unimod.*(rand([neurons, m+1]))};
for layers = 2:n_layers-2
    w(layers) = {-unimod+2*unimod.*(rand(neurons,neurons+1))};
end
w(n_layers-1) = {-unimod+2*unimod.*(rand([y, neurons+1]))};



switch mode
    case 'mini-batch'
        batchsize = 25;
    case 'batch'
    	batchsize = sizedata;
	case 'stochastic'
    	batchsize = 1;
    otherwise
        batchsize = sizedata;
        warning(sprintf('%s is not a supported batch mode, switching to full batch.',mode))
end


eta_orig=eta;
for epoch = 1:iterations
    out = zeros(size(data,1),y);
    i_data = randperm(sizedata);
    for batch=1:floor(sizedata/batchsize);
        points=(batch-1)*batchsize+1:(batch)*batchsize;
        a=cell([n_layers-1, 1]);
        z=cell([n_layers, 1]);
        delta=z;
        if dropout
            z{1}= [data(i_data(points),1:2).*(round(1-rand(size(data(i_data(points),1:2))).^2)) ones(batchsize,1)];
        else
            z{1}= [data(i_data(points),1:2) ones(batchsize,1)];
        end
        

%% Forward propagate 
        for layer = 1:n_layers-1
            a{layer} = z{layer}*w{layer}';
            z{layer+1}=[activator(a{layer},type) ones(batchsize,1)];
        end

        z{n_layers}=softmax(a{end}')';

%% Evaluate Delta k for output layer
        delta{n_layers} = z{n_layers}-data(i_data(points),3:4);

%% Backpropagate Delta j for hidden layers
        for hidden = n_layers-1:-1:2
            delta{hidden} = delta{hidden+1} * w{hidden}(:,1:end-1) .* diffact(a{hidden-1},type);
        end
%% Evaluate Derivatives
        for layer=1:n_layers-1
            deriv_E_w{layer} = delta{layer+1}'*z{layer};
            w{layer} = w{layer}-eta*deriv_E_w{layer};
        end
    end
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
a=cell([n_layers-1, 1]);
z=cell([n_layers, 1]);
delta=z;
z{1}= [data(:,1:2) ones(sizedata,1)];
for layer = 1:n_layers-1
    a{layer} = z{layer}*w{layer}';
	z{layer+1}=[activator(a{layer},type) ones(sizedata,1)];
end
z{n_layers}=softmax(a{end}')';
out=z{end};

display(sprintf('The total missclassification in the final run is %0.2f%%.',100-sum(round(out(:,2)) == data(:,4))/sizedata*100))


%% Plot
figure
id = find(out(:,1)>.5);
plot(data(id,1), data(id,2), 'b.', 'MarkerSize', 20);
hold on
id = find(out(:,2)>.5);
plot(data(id,1), data(id,2), 'r.', 'MarkerSize', 20);
id = find(round(out(:,1)) ~= data(:,3));
plot(data(id,1), data(id,2), 'g.', 'MarkerSize', 5);
id = find(round(out(:,2)) ~= data(:,4));
plot(data(id,1), data(id,2), 'g.', 'MarkerSize', 5);
legend('Class 1', 'Class 2', 'Missclassified')