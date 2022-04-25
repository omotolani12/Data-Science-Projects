function CW2()
close all;
clear all;
im0 = imread('Tolani2.jpg');
im0 = imresize(im0,0.20); % making the process faster
[M, N, dim] = size(im0)
im =  rgb2hsv(im0);%double(im0); %converting to HSV colour space
J = insertText(im0, [25 90 ], 'Omotolani');
   
figure(1),
subplot(2,4,1), imshow(J), title('RGB');
subplot(2,4,2), imshow(im), title('HSV');

% Converting an image into a 2D table with each row has RGB values
hs =[reshape(im(:,:,1),1,[]);...
        reshape(im(:,:,2),1,[]); %...   
        reshape(im(:,:,3),1,[])];

[dim, no] = size(hs)
X = hs';
X(1:10,:)

Y = zeros(no,1);

for i=1:no
       if X(i, 1)<0.2
           Y(i)=1;        % white pixel has ground truth 1 otherwise -1
       else
           Y(i)=-1;
       end
end

Y(1:10)
sz=[ size(Y)]


svm = fitcsvm(X,Y,'Standardize',true,'KernelFunction','rbf',...
                'KernelScale','auto');
sz = size(svm)
cv = crossval(svm)
loss = kfoldLoss(cv)
[~, score] = kfoldPredict(cv);
sz_score = size(score);
mean(score<0)

predX = X;
 for i=1:no
    if score(i,2)<0
       predX(i,1:3)=0;
    end
 end

% reshape back to image resolution (50x50)
im_pred(:,:,1) = reshape(predX(:,1),M,N);
im_pred(:,:,2) = reshape(predX(:,2),M,N);
im_pred(:,:,3) = reshape(predX(:,3),M,N);
im_pred_rgb = hsv2rgb(im_pred);

subplot(2,4,3), imshow(mat2gray(im_pred)), title('Predicted HSV');
subplot(2,4,4), imshow(mat2gray(im_pred_rgb)), title('Predicted RGB');
        
%display edges
im_eg = rgb2gray(im_pred_rgb);
subplot(2,4,5), imshow(edge(im_eg, 'sobel')); title('Edge =Sobel');
subplot(2,4,6), imshow(edge(im_eg, 'canny')); title('Edge =Canny');
subplot(2,4,7), imshow(edge(im_eg, 'roberts')); title('Edge =Roberts');
subplot(2,4,8), imshow(edge(im_eg, 'log')); title('Edge =Log');


