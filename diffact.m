function out=diffact(in,name) % derivative of activation function
    switch name
        case 'relu'
            out=(in>0);
        case 'sigmoid'
            out=in.*(1-in);
        case 'tanh'
            out=1-in.^2;
        otherwise
            out=(in>0);
    end
end