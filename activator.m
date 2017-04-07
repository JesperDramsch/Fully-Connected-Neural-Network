function out=activator(in,name)
    switch name
        case 'relu'
            out=max(in,0);
        case 'sigmoid'
            out=1./(1+exp(-in));
        case 'tanh'
            out=tanh(in);
    end
end