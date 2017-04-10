function out=activator(in,name)
    switch name
        case 'relu'
            out=max(in,0);
        case 'sigmoid'
            out=1./(1+exp(-in));
        case 'tanh'
            out=tanh(in);
        otherwise
            warning(sprintf('%s is not a supported activator type, switching to RelU.',name))
            out=max(in,0);
    end
end