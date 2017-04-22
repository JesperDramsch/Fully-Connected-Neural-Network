function check_gradients(in,labels,w,n_layers,activator_type)
    eps = 1e-5;
    sizedata = size(in,1);
    z = forward_NN([in ones(sizedata ,1)],w,n_layers,activator_type);
	deriv_E_w = backward_NN(z,labels,w,n_layers,activator_type);
    for layer=1:n_layers-1
        for j = 1:size(w{layer},1)
            for k = 1:size(w{layer},2)

                grad1 = deriv_E_w{layer}(j,k);
                
                w{layer}(j,k) = w{layer}(j,k)-eps;
                y = forward_NN([in ones(sizedata ,1)],w,n_layers,activator_type);
                cost1 = costfunction(y{end},labels,'Cross-entropy');
                
                w{layer}(j,k) = w{layer}(j,k)+2*eps;
                y = forward_NN([in ones(sizedata ,1)],w,n_layers,activator_type);
                cost2 = costfunction(y{end},labels,'Cross-entropy');

                w{layer}(j,k) = w{layer}(j,k)-eps;

                grad2 = (cost2 - cost1) / (2*eps);

                assert(abs(grad1-grad2)<=eps)
            end
        end
    end
    display('Gradient Test Ok')