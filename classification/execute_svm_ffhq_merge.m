clear all; clc;

addpath './libsvm-mat-3.0-3'
% C:\000_NaZhangBack\1_deepfake\3_SVM

% item_list = {'1_sseq','2_brisque','5_curvelet','6_niqe','7_tmiqa'}


%% 1_sseq
train_f1 = []
train_l1 = []
test_f1 = []
test_l1 = []
% train
load './feature/1_sseq/FFHQ_small_train_data.mat'
train_f1 = features
train_l1 = labels
% test
load './feature/1_sseq/FFHQ_small_test_data.mat'
test_f1 = features
test_l1 = labels


%% 2_brisque
train_f2 = []
train_l2 = []
test_f2 = []
test_l2 = []
% train
load './feature/2_brisque/FFHQ_small_train_data.mat'
train_f2 = features
train_l2 = labels
% test
load './feature/2_brisque/FFHQ_small_test_data.mat'
test_f2 = features
test_l2 = labels


%% 5_curvelet
train_f5 = []
train_l5 = []
test_f5 = []
test_l5 = []
% train
load './feature/5_curvelet/FFHQ_small_train_data.mat'
train_f5 = features
train_l5 = labels
% test
load './feature/5_curvelet/FFHQ_small_test_data.mat'
test_f5 = features
test_l5 = labels


%% 6_niqe
train_f6 = []
train_l6 = []
test_f6 = []
test_l6 = []
% train
load './feature/6_niqe/FFHQ_small_train_data.mat'
train_f6 = features
train_l6 = labels
% test
load './feature/6_niqe/FFHQ_small_test_data.mat'
test_f6 = features
test_l6 = labels


%% 7_tmiqa
train_f7 = []
train_l7 = []
test_f7 = []
test_l7 = []
% train
load './feature/7_tmiqa/FFHQ_small_train_data.mat'
train_f7 = features
train_l7 = labels
% test
load './feature/7_tmiqa/FFHQ_small_test_data.mat'
test_f7 = features
test_l7 = labels


%% % Train the SVM Classifier

train_la = train_l1 % [train_l1, train_l2, train_l5, train_l6, train_l7]
test_la = test_l1 %[test_l1, test_l2, test_l5, test_l6, test_l7]
train_fa = [train_f6, train_f7]
test_fa = [test_f6, test_f7]

% train_fa = [train_f1,  train_f2, train_f5, train_f6, train_f7]
% test_fa = [test_f1, test_f2, test_f5, test_f6, test_f7]


svm_18 = fitcsvm(train_fa,train_la);
[test_label,score] = predict(svm_18, test_fa);
count = 0
len = size(test_label,1)
for i=1:len
    if test_la(i,1) == test_label(i,1)
        count = count + 1
    end
end
accuracy_liner = count/len
% save('ff8k_svm_liner_label.mat', 'label')


% train svm: rbf
svm_19 = fitcsvm(train_fa,train_la,'Standardize',true,'KernelFunction','rbf');
[test_label9,score9] = predict(svm_19, test_fa);
count9 = 0
len9 = size(test_label9,1)
for i=1:len9
    if test_la(i,1) == test_label9(i,1)
        count9 = count9 + 1
    end
end
accuracy_rbf = count9/len9
% save('ff8k_svm_rbf_label.mat', 'label')

% model = svmtrain(train_feature_18,train_label);
% [predict_label, accuracy, dec_values] = svmpredict(test_label, test_feature_18, model);



