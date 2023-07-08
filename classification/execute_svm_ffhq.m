clear all; clc;

addpath './libsvm-mat-3.0-3'
% C:\000_NaZhangBack\1_deepfake\3_SVM

item_list = {'1_sseq','2_brisque','5_curvelet','6_niqe','7_tmiqa'}

train_f = []
train_l = []
test_f = []
test_l = []

% train
load './feature/7_tmiqa/FFHQ_small_train_data.mat'
train_f = features
train_l = labels

% test
load './feature/7_tmiqa/FFHQ_small_test_data.mat'
test_f = features
test_l = labels

% Train the SVM Classifier
svm_18 = fitcsvm(train_f,train_l);
[test_label,score] = predict(svm_18, test_f);
count = 0
len = size(test_l,1)
for i=1:len
    if test_l(i,1) == test_label(i,1)
        count = count + 1
    end
end
accuracy_liner = count/len
% save('ff8k_svm_liner_label.mat', 'label')


% train svm: rbf
svm_18 = fitcsvm(train_f,train_l,'Standardize',true,'KernelFunction','rbf');
[test_label,score] = predict(svm_18, test_f);
count = 0
len = size(test_l,1)
for i=1:len
    if test_l(i,1) == test_label(i,1)
        count = count + 1
    end
end
accuracy_rbf = count/len
% save('ff8k_svm_rbf_label.mat', 'label')

% model = svmtrain(train_feature_18,train_label);
% [predict_label, accuracy, dec_values] = svmpredict(test_label, test_feature_18, model);



