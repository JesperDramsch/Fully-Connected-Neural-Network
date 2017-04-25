close all
clear all
clc;
format compact

train = load('MNIST.mat');
data = pca(double(train.data)');
labels = double(train.label);


indexes = randi([1 6000],1,5000);
data = double(train.data(indexes,:));
labels = double(train.label(indexes,:));

iterations = 1000;
n_layers = 5; %number of layers
neurons = 125; %+bias
y = min(size(labels)); %number of outputs
cnn_layers = 1;
cnn_kernels= 1;
eta = 1e-5; % Learning Rate
acttype = 'relu' % sigmoid, tanh or relu
mode = 'batch' % batch, mini-batch or stochastic
training = true;
dropout = true; % Dropout flag
dropout_val = [0 0 .2 0];


%% Condition MNIST
[~,max_i_mnist] = max(labels,[],2);


%% stuff
m = 196; %number of inputs
w=weights_NN(m,y,neurons,n_layers);
delta_w=cellfun(@(x) x*0,w,'un',0);
im_data=permute(reshape(data,[],28,28),[2 3 1]);
sizedata = max(size(data));
best_cost = inf(1);
filters = filters_NN(3,3,cnn_kernels,cnn_layers);

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

counter=0;

%% loop

for epoch = 1:iterations
    i_data = randperm(sizedata);
    %% CNN
    
    %% FC
    for batch=1:floor(sizedata/batchsize)
        points=(batch-1)*batchsize+1:(batch)*batchsize;
        %reshape(conv2(reshape(data(10,:),[28 28]),k,'same'),[1 784])
        cnn_layer=1;
        % convolution
        for ff = 1:cnn_kernels
            z_cnn=activator(imfilter(im_data(:,:,i_data),filters{1}(:,:,ff),'conv'),acttype);
        end
        
        % maxpooling
        if mod(size(z_cnn,1),2)~=0
            z_cnn=padarray(z_cnn,[1 1],0,'post');
        end
        z_size=size(z_cnn);
        
        maxpooled=zeros(z_size./[2 2 1]);
        winner=maxpooled;
        for xx= 1:2:z_size(1)
            for yy = 1:2:z_size(2)
                [maxpooled((xx+1)/2,(yy+1)/2,:,:), winner((xx+1)/2,(yy+1)/2,:,:)]=max([z_cnn(xx,yy,:,:) z_cnn(xx+1,yy,:,:) z_cnn(xx,yy+1,:,:) z_cnn(xx+1,yy+1,:,:)]);
            end
        end
        proc_data=maxpooled;
        
        
        maxpooled=reshape(ipermute(maxpooled,[2 3 1]),[],14*14);
        %z_cnn=reshape(ipermute(z_cnn,[2 3 1]),[],28*28);
        
        
        %% Forward feed
        z = forward_NN([maxpooled(i_data(points),:) ones(batchsize,1)],w,n_layers,acttype,training,dropout,dropout_val);

        
        %% Backward propagate
        delta=cell([n_layers, 1]);
        deriv_E_w=cell([n_layers-1, 1]);
        % Evaluate Delta k for output layer
        delta{n_layers} = z{n_layers}-labels(i_data(points),:);
        %% Backpropagate Delta j for hidden layers
        for layer = n_layers-1:-1:1
            deriv_E_w{layer} = z{layer}'*delta{layer+1};
            
            delta{layer} = (delta{layer+1} * w{layer}(1:end-1,:)') .* diffact(z{layer}(:,1:end-1),acttype);
        end
        
        %% Evaluate Derivatives
        new_w = update_NN(w,delta_w,deriv_E_w,n_layers,eta);
        delta_w = cellfun(@minus,w,new_w,'Un',0);
        w=new_w;
        
        
        %% Backpropagate CNN
               
        
        tmp_delta=permute(reshape(delta{1},[],14,14),[2 3 1]);
        cnn_delta=zeros(28,28,batchsize);
        for xx= 1:size(tmp_delta,1)
            for yy = 1:size(tmp_delta,2)
                switch winner(xx,yy)
                    case 1
                        cnn_delta(xx*2-1,yy*2-1,:,:) = tmp_delta(xx,yy,:,:);
                    case 2
                        cnn_delta(xx*2,yy*2-1,:,:) = tmp_delta(xx,yy,:,:);
                    case 3
                        cnn_delta(xx*2-1,yy*2,:,:) = tmp_delta(xx,yy,:,:);
                    case 4
                        cnn_delta(xx*2,yy*2,:,:) = tmp_delta(xx,yy,:,:);
                end
            end
        end
        %cnn_delta=ra
        
        % Backpropagate Delta j for hidden layers

        %cnn_deriv_E_w = imfilter(z_cnn,cnn_delta,'conv');
        
        %cnn_delta = (cnn_delta * w(1:end-1,:)') .* diffact(maxpooled(:,1:end-1),acttype);
        
        for ff = 1:cnn_kernels
            cnn_delta2=reshape(ipermute(imfilter(permute(reshape(cnn_delta,[],28,28),[2 3 1]),filters{1}(:,:,ff)).*permute(reshape(diffact(z_cnn,acttype),[],28,28),[2 3 1]),[2 3 1]),[],28*28);
            filters{ff} = filters{ff}-eta*(cnn_delta'*cnn_delta2);
        end
    end
    
    %% Validation
    randval=indexes;
    %randval = randi([1 60000],5000,1);
    validate = double(train.data(randval,:));
    validate_label = train.label(randval,:);
    z = forward_NN([validate ones(length(validate(:,1)),1)],w,n_layers,acttype,false,dropout,dropout_val);
    cost= costfunction(z{end},validate_label,'RMS');
    totalerror=mnist_error(max_i_mnist(:),z{end});
    %totalerror=mnist_error(max_i_mnist(randval),z{end});
    fprintf('Iteration: %i, Cost: %f; Error: %f%%\n',epoch, cost, totalerror)
    if cost < best_cost
        best_cost = cost;
        counter = 0;
        backup_w = w;
    else
        counter = counter +1;
        if counter > iterations*.35
            break
        end
        
    end
    w=backup_w;
end
%% Wrap it up
indexes = randi([1 6000],1,sizedata);
data = double(train.data(indexes,:));
labels = double(train.label(indexes,:));

%% Test
out = forward_NN([data ones(sizedata,1)],w,n_layers,acttype,training,dropout,dropout_val);
out=out{end};

save('params.mat','w','n_layers','acttype','dropout','dropout_val')

%% Plots
%figure
%plotNN(m,neurons,y,n_layers)

