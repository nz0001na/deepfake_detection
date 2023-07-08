clear all; clc;

addpath './libsvm-mat-3.0-3'
% C:\000_NaZhangBack\1_deepfake\3_SVM

item_list = {'1_sseq','2_brisque','5_curvelet','6_niqe','7_tmiqa'}
for t=3:length(item_list)
    pp = item_list{t}

    src_path = ['C:\000_NaZhangBack\1_deepfake\NNS_output\ff_data_4000\' pp '\']


    f_item = {'Deepfakes','Face2Face','FaceSwap','NeuralTextures'};
    for i =1:1%length(f_item)
        result = []
        train_f = []
        train_l = []
        test_f = []
        test_l = []
        src_real = [src_path 'Raw/']
    % real
        sub_list = dir(src_real)
        sub_list=sub_list(~ismember({sub_list.name},{'.','..'}));
        % count = length(sub_list)
        for n =1:500
            load([src_real sub_list(n,1).name ])
            train_f = [train_f; train_feature]
            train_l = [train_l; train_label]
        end

        for m =501:1000
            load([src_real sub_list(m,1).name ])
            test_f = [test_f; train_feature]
            test_l = [test_l; train_label]
        end

    %     fake
        sub_list1 = dir([src_path f_item{i}])
        sub_list1=sub_list1(~ismember({sub_list1.name},{'.','..'}));
        num = length(sub_list1)

        for n =1:500
            load([src_path f_item{i} '/' sub_list1(n,1).name ])
            train_f = [train_f; train_feature]
            train_l = [train_l; train_label]
        end

        for m =501:1000
            load([src_path f_item{i} '/' sub_list1(m,1).name ])
            test_f = [test_f; train_feature]
            test_l = [test_l; train_label]
        end  

        % Train the SVM Classifier
        svm_18 = fitcsvm(train_f,train_l);
        [test_label,score] = predict(svm_18, test_f);
        count = 0
        len = size(test_l,1)
        for q=1:len
            if test_l(q,1) == test_label(q,1)
                count = count + 1
            end
        end
        accuracy_liner = count/len
        % save('ff8k_svm_liner_label.mat', 'label')
    %     disp([f_item{i} ': ' num2str(accuracy_liner)])
        result = [result; accuracy_liner]

        % train svm: rbf
        svm_18 = fitcsvm(train_f,train_l,'Standardize',true,'KernelFunction','rbf');
        [test_label,score] = predict(svm_18, test_f);
        count = 0
        len = size(test_l,1)
        for q=1:len
            if test_l(q,1) == test_label(q,1)
                count = count + 1
            end
        end
        accuracy_rbf = count/len
    %      disp([f_item{i} ': ' num2str(accuracy_rbf)])
        result = [result; accuracy_rbf]
        
        save([  pp '_' f_item{i} '_result.mat'],'result') 




    end
    end

% 
% % Train the SVM Classifier
% svm_18 = fitcsvm(train_f,train_l);
% [test_label,score] = predict(svm_18, test_f);
% count = 0
% len = size(test_l,1)
% for i=1:len
%     if test_l(i,1) == test_label(i,1)
%         count = count + 1
%     end
% end
% accuracy_liner = count/len
% % save('ff8k_svm_liner_label.mat', 'label')
% 
% 
% % train svm: rbf
% svm_18 = fitcsvm(train_f,train_l,'Standardize',true,'KernelFunction','rbf');
% [test_label,score] = predict(svm_18, test_f);
% count = 0
% len = size(test_l,1)
% for i=1:len
%     if test_l(i,1) == test_label(i,1)
%         count = count + 1
%     end
% end
% accuracy_rbf = count/len


