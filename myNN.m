close all;
clear all;
clc;
format compact

addpath('..')

iterations = 10;
neurons = 4; %+bias
m = 2; %number of inputs
y = 2; %number of outputs
type = 'relu' % sigmoid, tanh or relu
mode = 'mini-batch' % batch, mini-batch or stochastic

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
        batchsize = 10;
    case 'batch'
    	batchsize = sizedata;
	case 'stochastic'
    	batchsize = 1;
    otherwise
        batchsize = sizedata;
        warning(sprintf('%s is not a supported batch mode, switching to full batch.',mode))
end

figure
for epoch = 1:iterations
    out = zeros(size(data,1),y);
    for batch=1:floor(sizedata/batchsize);
        points=(batch-1)*batchsize+1:(batch)*batchsize;
        a=cell([n_layers-1, 1]);
        z=cell([n_layers, 1]);
        delta=z;
        z{1}= [data(points,1:2) ones(batchsize,1)];

%% Forward propagate 
        for layer = 1:n_layers-1
            a{layer} = z{layer}*w{layer}';
            z{layer+1}=[activator(a{layer},type) ones(batchsize,1)];
        end

        z{n_layers}=softmax(a{end}')';

%% Evaluate Delta k for output layer
        delta{n_layers} = z{n_layers}-data(points,3:4);

%% Backpropagate Delta j for hidden layers
        for hidden = n_layers-1:-1:2
            delta{hidden} = delta{hidden+1} * w{hidden}(:,1:end-1) .* diffact(a{hidden-1},type);
        end
%% Evaluate Derivatives
        for layer=1:n_layers-1
            deriv_E_w{layer} = delta{layer+1}'*z{layer};
            w{layer} = w{layer}-eta*deriv_E_w{layer};
        end
        
        out(points,:) = z{end};
    end
%% Error calculation
    
    bar(epoch,sum(round(out(:,2)) == data(:,4))/sizedata*100)
    ylim([0 100])
    hold on
end

a=cell([n_layers-1, 1]);
z=cell([n_layers, 1]);
delta=z;
        z{1}= [data(:,1:2) ones(sizedata,1)];

%% Forward propagate 
for layer = 1:n_layers-1
    a{layer} = z{layer}*w{layer}';
	z{layer+1}=[activator(a{layer},type) ones(sizedata,1)];
end

z{n_layers}=softmax(a{end}')';

display(sprintf('The total missclassification in the final run is %0.2f%%.',100-sum(round(out(:,2)) == data(:,4))/sizedata*100))

hold off
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