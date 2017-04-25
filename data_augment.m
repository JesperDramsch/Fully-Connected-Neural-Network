train = load('MNIST.mat');
data = double(train.data);
label = double(train.label);

% Elastic Distortions on Stackexchange http://stackoverflow.com/questions/39308301/expand-mnist-elastic-deformations-matlab

%data=data(1:30,:);

data=permute(reshape(data,[],28,28),[2 3 1]);
nummer=randi([1 60000]);

datapoints=max(size(data));
i_data = randperm(datapoints);

label=[label; label(i_data,:)];

new=data.*0;
for layer=1:datapoints
    img=data(:,:,i_data(layer));
dx = -1+2*rand(size(img)); 
dy = -1+2*rand(size(img)); 

sig=4; 
alpha=60;
H=fspecial('gauss',[7 7], sig);
fdx=imfilter(dx,H);
fdy=imfilter(dy,H);
n=sum((fdx(:).^2+fdy(:).^2)); %// norm (?) not quite sure about this "norm"
fdx=alpha*fdx./n;
fdy=alpha*fdy./n;

[y, x]=ndgrid(1:size(img,1),1:size(img,2));

new(:,:,layer) = griddata(x-fdx,y-fdy,double(img),x,y);
end
new(isnan(new))=0;

figure;
imagesc(img(:,:,end)); colormap gray; axis image; axis tight;
hold on;
quiver(x,y,fdx(:,:),fdy(:,:),0,'r');

figure;
subplot(121); imagesc(img); axis image;
subplot(122); imagesc(new(:,:,end)); axis image;
colormap gray

data=[train.data; reshape(ipermute(new,[2 3 1]),[],28*28)];

save('MNIST_train.mat','data','label')