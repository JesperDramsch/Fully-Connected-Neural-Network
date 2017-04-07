function out=diffact(in,name) % derivative of activation function
    switch name
        case 'relu'
            out=(in>0);
        case 'sigmoid'
            out=activator(in,name).*(1-activator(in,name));
        case 'tanh'
            out=1-activator(in,name).^2;
    end
end