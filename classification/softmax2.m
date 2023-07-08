clear all; clc;


train_feature_18 = csvread('train_feature_18_4000.csv')
train_label = csvread('train_label_4000.csv')
train_label = train_label'
train_Y = []
n = size(train_label,1)
for i=1:n
    lab = train_label(i)
    if lab == 0
        train_Y = [train_Y; [1,0]]
    else
        train_Y = [train_Y; [0,1]]
    end
%     print('d')
end


test_feature_18 = csvread('test_feature_18_all.csv')
test_label = csvread('test_label_all.csv')
test_label = test_label'
test_Y = []
n = size(test_label,1)
for i=1:n
    lab = test_label(i)
    if lab == 0
        test_Y = [test_Y; [1,0]]
    else
        test_Y = [test_Y; [0,1]]
    end
%     print('d')
end

X = train_feature_18'
Y = train_Y'
X1 = test_feature_18'
Y1 = test_Y'
% softmax
net = trainSoftmaxLayer(X,Y);
Y0 = net(X1);
plotconfusion(Y1,Y0);

