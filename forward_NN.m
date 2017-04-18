function z = forward_NN(in,w,n_layers,activator_type,training)
	a=cell([n_layers-1, 1]);
    z=cell([n_layers, 1]);
    z{1}=in;
    
	for layer = 1:n_layers-1
    	a{layer} = z{layer}*w{layer};
        z{layer+1}=[activator(a{layer},activator_type) ones(size(in,1),1)];
	end

    z{n_layers}=softmax(a{end}')';

