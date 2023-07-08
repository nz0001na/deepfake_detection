clear all; clc;


% addpath(genpath(pwd))

addpath './libsvm-mat-3.0-3'


train_feature_18 = csvread('train_feature_18_4000.csv')
train_label = csvread('train_label_4000.csv')
train_label = train_label'

% train_Y = []
% n = size(train_label,1)
% for i=1:n
%     lab = train_label(i)
%     if lab == 0
%         train_Y = [train_Y; [1,0]]
%     else
%         train_Y = [train_Y; [0,1]]
%     end
% %     print('d')
% end

test_feature_18 = csvread('test_feature_18_all.csv')
test_label = csvread('test_label_all.csv')
test_label = test_label'
% test_Y = []
% n = size(test_label,1)
% for i=1:n
%     lab = test_label(i)
%     if lab == 0
%         test_Y = [test_Y; [1,0]]
%     else
%         test_Y = [test_Y; [0,1]]
%     end
% %     print('d')
% end




%Train the SVM Classifier
% svm_18 = fitcsvm(train_feature_18,train_label,'KernelFunction','linear');
% [label,score] = predict(svm_18, test_feature_18);

nc=[0.0001,0.1,0.5,1,5,10,15,20,25,30,35,40,45,50,70,100];

ft = fopen('18_predict_result.csv', 'w');
fprintf(ft, '%s,%s\n', 'cmd', 'accuracy');
        
% model_face=svmtrain(train_label, train_feature_18);
% [label_result, accuracy_result, dec_test_result]=svmpredict(test_label,test_feature_18, model_face);

% SVM does Classification  16 
for ci=1:16
    for log2g = -10:0.1:10
        cmd = ['-c ', num2str(nc(ci)), ' -g ', num2str(2^log2g)];
            disp(cmd) 
        model = svmtrain(train_label, train_feature_18,'-t 1');
        [predict_label, accuracy, dec_values] = svmpredict(test_label, test_feature_18, model);
        % The second output, accuracy, is a vector including accuracy (for classification),
        % mean squared error, and squared correlation coefficient (for regression).
        fprintf(ft, '%s,%f\n', cmd, accuracy(1));
%         printf('%s,%f\n', cmd, accuracy(1));
    end
end

fclose(ft)




