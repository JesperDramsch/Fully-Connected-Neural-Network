function cost= costfunction(in,labels,cost_type)
    switch cost_type
        case 'Cross-entropy'
            cost = mean(-sum(labels.*log(in)+ (1-labels).*log(1-in)));
        case 'RMS'
            cost = 0.5 .* mean(sum((labels - in).^2));
        otherwise
            cost = mean(-sum(labels.*log(in)+ (1-labels).*log(1-in)));
            warning('Switching to Cross-entropy')
    end