function w = update_NN(w,delta_w,deriv_E_w,n_layers,eta)
	for layer=1:n_layers-1
    	w{layer} = w{layer}-eta*deriv_E_w{layer}-4e-2*eta*w{layer}+delta_w{layer}*.95;
	end