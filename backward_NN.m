function deriv_E_w = backward_NN(z,label,w,n_layers,activator_type)
    delta=cell([n_layers, 1]);
    deriv_E_w=cell([n_layers-1, 1]);
%% Evaluate Delta k for output layer     
    delta{n_layers} = z{n_layers}-label;
%% Backpropagate Delta j for hidden layers
    for layer = n_layers-1:-1:1
        deriv_E_w{layer} = z{layer}'*delta{layer+1};
        
    	delta{layer} = (delta{layer+1} * w{layer}(1:end-1,:)') .* diffact(z{layer}(:,1:end-1),activator_type);
    end
