[X,T] = iris_dataset;

% X is a 4x150 matrix of four attributes of iris flowers: Sepal length, sepal width, petal length, petal width.
% 
% T is a 3x150 matrix of associated class vectors defining which of the three classes each input is assigned to. 
% Each row corresponds to a dummy variable representing one of the iris species (classes). 
% In each column, a 1 in one of the three rows represents the class that particular sample 
% (observation or example) belongs to. There is a zero in the rows for the other classes that the observation 
% does not belong to.
% 
% Train a softmax layer using the sample data.

net = trainSoftmaxLayer(X,T);

% Classify the observations into one of the three classes using the trained softmax layer.

Y = net(X);

% Plot the confusion matrix using the targets and the classifications obtained from the softmax layer.

plotconfusion(T,Y);