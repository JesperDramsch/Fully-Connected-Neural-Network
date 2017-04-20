function totalerror = mnist_error(mnist_ind,labels)
    [~,l_ind]= max(labels,[],2);
    right=(l_ind==mnist_ind);
    totalerror = 100*(1-sum(right)/length(mnist_ind));