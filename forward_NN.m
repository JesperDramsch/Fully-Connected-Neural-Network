function z = forward_NN(in,w,n_layers,activator_type,training,dropout,dropout_in,dropout_hidden)

	a=cell([n_layers-1, 1]);
    z=cell([n_layers, 1]);
    z{1}=in;
    
	if not(dropout)
        dropout_in=1;
    	dropout_hidden=1;
	end
    if training
        for layer = 1:n_layers-1
            if layer == 1
                drop_mod = dropout_in;
            elseif layer == n_layers-1
                drop_mod = 1;
            else
                drop_mod = dropout_hidden;
            end
            a{layer} = (z{layer}.*random('bino',1,drop_mod,size(z{layer})))*w{layer};
            
            z{layer+1}=[activator(a{layer},activator_type) ones(size(in,1),1)];
        end
    else
        for layer = 1:n_layers-1
            if layer == 1
                drop_mod = 1;
            elseif layer == n_layers-1
                drop_mod = 1;
            else
                drop_mod = dropout_hidden;
            end
            a{layer} = z{layer}*w{layer}*drop_mod;
            z{layer+1}=[activator(a{layer},activator_type) ones(size(in,1),1)];
        end
    end
    
    z{n_layers}=softmax(a{end}')';

