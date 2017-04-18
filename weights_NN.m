function w = weights_NN(m,y,neurons,n_layers)
    unimod = 1*m^-.5;
    w = cell([n_layers-1, 1]);
    w(1) = {-unimod+(unimod+unimod)*(rand([m+1, neurons]))};
    for layers = 2:n_layers-2
        w(layers) = {-unimod+(unimod+unimod)*(rand(neurons+1,neurons))};
    end
    w(n_layers-1) = {-unimod+(unimod+unimod)*(rand([neurons+1,y]))};
