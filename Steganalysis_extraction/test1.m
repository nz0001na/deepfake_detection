
img_path = '/home/guo/1_Nazhang/1_deepfake/1_code/steg_code/crop_face/'

% ft = fopen('ppp.csv', 'w');
% fprintf(ft, '%s\n', 'i');


train_feature = []
train_label = []
data = fopen('2Forensics_partial_train_list.csv');
A = textscan(data,'%s','Delimiter','\n');
B = A{1,1};
len = size(B,1)
for i=1227+1:len
%    fprintf(ft, '%s\n', i);

   C = B{i,1}
   c1 = split(C,",")
   path = split(c1{1,1},"/")
   id = path{9,1}
   name = path{10,1}
   label = str2num(c1{2,1})
   img = [img_path id '/' name];
   a = exist(img)
   if a == 0
       continue
   end
   
   f = SRM(img)
        
   fields = fieldnames(f);
   len = size(fields,1)

   final_feature = []
   for m=1:len
       field = fields{m}
       value = getfield(f,field)
       final_feature = [final_feature,value]
    end
   
   train_feature = [train_feature; final_feature]
   train_label = [train_label;label]
    
end

save('train_data_1227_.mat','train_feature','train_label') 


% % test
% test_feature = []
% test_label = []
% data = fopen('2Forensics_partial_test_list.csv');
% A = textscan(data,'%s','Delimiter','\n');
% B = A{1,1};
% len = size(B,1)
% for i=1:len
%    fprintf(ft, '%s\n', i);
%    C = B{i,1}
%    c1 = split(C,",")
%    path = split(c1{1,1},"/")
%    id = path{9,1}
%    name = path{10,1}
%    label = str2num(c1{2,1})
%    img = [img_path id '/' name];
%    a = exist(img)
%    if a == 0
%        continue
%    end
%    f = SRM(img)
%         
%    fields = fieldnames(f);
%    len = size(fields,1)
% 
%    final_feature = []
%    for m=1:len
%        field = fields{m}
%        value = getfield(f,field)
%        final_feature = [final_feature,value]
%     end
%    
%    test_feature = [test_feature; final_feature]
%    test_label = [test_label;label]
%     
% end
% 
% save('test_data.mat','test_feature','test_label') 
% 
% % SVM
% model = svmtrain(train_label, train_feature, 'kernel_function','rbf');
% [predict_label, accuracy, dec_values] = svmpredict(test_label, test_feature, model);
% fprintf(ft,'rbf : %s,%f\n', cmd, accuracy(1));
% 
% 
% % SVM
% model = svmtrain(train_label, train_feature, 'kernel_function','linear');
% [predict_label, accuracy, dec_values] = svmpredict(test_label, test_feature, model);
% fprintf(ft,'linear : %s,%f\n', cmd, accuracy(1));
% 
% 
% % SVM
% model = svmtrain(train_label, train_feature, 'kernel_function','quadratic');
% [predict_label, accuracy, dec_values] = svmpredict(test_label, test_feature, model);
% fprintf(ft,'quadratic : %s,%f\n', cmd, accuracy(1));
% 
% 
% % SVM
% model = svmtrain(train_label, train_feature, 'kernel_function','polynomial');
% [predict_label, accuracy, dec_values] = svmpredict(test_label, test_feature, model);
% fprintf(ft,'polynomial : %s,%f\n', cmd, accuracy(1));
% 
% 
% 
% % SVM
% model = svmtrain(train_label, train_feature, 'kernel_function','mlp');
% [predict_label, accuracy, dec_values] = svmpredict(test_label, test_feature, model);
% fprintf(ft,'mlp : %s,%f\n', cmd, accuracy(1));


% fclose(ft)



















