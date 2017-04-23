function z = forward_NN(in,w,n_layers,activator_type,training,dropout,dropout_val)

	a=cell([n_layers-1, 1]);
    z=cell([n_layers, 1]);
    z{1}=in;
    
	if not(dropout)
        dropout_val=zeros(1,n_layers-1);
    end
    for layer = 1:n_layers-1
        if training
            a{layer} = (z{layer}.*(1/(1-dropout_val(layer))).*random('bino',1,1-dropout_val(layer),size(z{layer})))*w{layer};
        else
            a{layer} = z{layer}*w{layer};
        end
        z{layer+1}=[activator(a{layer},activator_type) ones(size(in,1),1)];
    end
    
    z{n_layers}=softmax(a{end}')';

