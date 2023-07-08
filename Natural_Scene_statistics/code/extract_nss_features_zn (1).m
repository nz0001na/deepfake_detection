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
for n=1:length(f_item)
    item = f_item{n};
    if ~exist([dst_path item '/'], 'dir')
        mkdir([dst_path item '/'])
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
        features =[]
        labels = []
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
           f = call_feature_function_zn(img,n)

           features = [features; f]
           labels = [labels;label]
        end
        
       save([dst_path item '/' 'FFHQ_small_' set '_data.mat'],'features','labels') 
    end
   
   
end


