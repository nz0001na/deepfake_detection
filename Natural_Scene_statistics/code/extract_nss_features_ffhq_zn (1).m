clear;clc;
% add path to external functions
addpath('../lib/redist-source');
addpath('../lib/bovik/SSEQ_release/SSEQ');
addpath('../lib/bovik/BLIINDS2_release/Bliinds2_code');
addpath('../lib/bovik/DIIVINE_release/');
addpath('../lib/bovik/DIIVINE_release/sepspyr/deps/matlabPyrTools-1.3');
addpath('../lib/bovik/DIIVINE_release/sepspyr/deps/matlabPyrTools-1.3/MEX');
addpath('../lib/bovik/BRISQUE_release/BRISQUE_release');
addpath('../lib/bovik/CurveletQA_release/');
addpath('../lib/bovik/CurveletQA_release/fdct_usfft_matlab');
addpath('../lib/bovik/CurveletQA_release/fdct_usfft_matlab/CurveCoeff');
addpath('../lib/bovik/CurveletQA_release/fdct_usfft_matlab/Utilities');
addpath('../lib/bovik/CurveletQA_release/fdct_usfft_matlab/Windows/Meyer');
addpath('../lib/bovik/CurveletQA_release/fdct_usfft_matlab/Windows/IteratedSine');
addpath('../lib/bovik/CurveletQA_release/fdct_usfft_matlab/USFFT');
addpath('../lib/bovik/niqe_release');
addpath('../lib/bovik/tmiqa_release');

% 
% sseqf = {}; bliindsf = {}; diivinef = {}; brisquef = {}; curveletf = {}; niqef = {}; tmiqa = {}; 

src_path = '/home/guo/Downloads/Data/FFHQ_1024_small/'
dst_path = '/home/guo/Downloads/Data/2_NNS_output/'

    
f_item = {'1_sseq','2_brisque','5_curvelet','6_niqe','7_tmiqa', '3_bliinds','4_diivine'};
for n=1:5%length(f_item)
    item = f_item{n};
%     if ~exist([dst_path num2str(299) '/' item '/'], 'dir')
%         mkdir([dst_path num2str(299) '/' item '/'])
%     end
%     if ~exist([dst_path num2str(400) '/' item '/'], 'dir')
%         mkdir([dst_path num2str(400) '/' item '/'])
%     end
%     if ~exist([dst_path num2str(600) '/' item '/'], 'dir')
%         mkdir([dst_path num2str(600) '/' item '/'])
%     end
    if ~exist([dst_path num2str(800) '/' item '/'], 'dir')
        mkdir([dst_path num2str(800) '/' item '/'])
    end

%     set_item = {'train','test'}
    for m=1:2
        if m == 1
            set = 'train'
            file = [src_path 'train_list_4000.csv'];
        else
            set = 'test'
            file = [src_path 'test_list_2000.csv'];
        end

    %     get list
        features1 =[]
        labels1 = []
        features2 =[]
        labels2 = []
        features3 =[]
        labels3 = []
        features4 =[]
        labels4 = []
        
        data = fopen(file);
        A = textscan(data,'%s','Delimiter','\n');
        B = A{1,1};
        len = size(B,1)
        for i=1:len
           C = B{i,1}
           c1 = split(C,",")
        %    path = split(c1{1,1},"/")
        %    id = path{9,1}
        %    name = path{10,1}
           label = str2num(c1{2,1})
           i
           img = [src_path c1{1,1}];
%            f1 = call_feature_function_ffhq_zn(img,n, 299)
%            f2 = call_feature_function_ffhq_zn(img,n, 400)
%            f3 = call_feature_function_ffhq_zn(img,n, 600)
           f4 = call_feature_function_ffhq_zn(img,n, 800)
    
%            features1 = [features1; f1]
%            labels1 = [labels1;label]
%            features2 = [features2; f2]
%            labels2 = [labels2;label]
%            features3 = [features3; f3]
%            labels3 = [labels3;label]
           features4 = [features4; f4]
           labels4 = [labels4;label]
        end

%        save([dst_path num2str(299) '/' item '/' 'FFHQ_small_' set '_data.mat'],'features1','labels1') 
%        save([dst_path num2str(400) '/' item '/' 'FFHQ_small_' set '_data.mat'],'features2','labels2') 
%        save([dst_path num2str(600) '/' item '/' 'FFHQ_small_' set '_data.mat'],'features3','labels3') 
       save([dst_path num2str(800) '/' item '/' 'FFHQ_small_' set '_data.mat'],'features4','labels4') 
    end
end


