load ionosphere;
Y = categorical(Y);
tabulate(Y);
xy = tsne(X, 'Verbose', 1)
scatter(xy(1:end,1), xy(1:end,2), [], Y, 'filled')
cvp = cvpartition(Y, 'Holdout', 0.3, 'Stratify', true);
X_train = X(training(cvp),:);
Y_train = Y(training(cvp),:);
X_test = X(test(cvp),:);
Y_test = Y(test(cvp),:);
svm = fitcsvm(X_train, Y_train, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
Y_pred = predict(svm, X_test)
accuracy = mean(Y_pred == Y_test)
confusionchart(Y_test, Y_pred)
% draw the support vectors
xy = tsne(X);
scatter(xy(1:end,1), xy(1:end,2), [], Y, 'filled')
hold on;
scatter(xy(svm.IsSupportVector,1), xy(svm.IsSupportVector,2), 62, 'r')