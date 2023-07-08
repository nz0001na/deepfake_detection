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

datatype_list = {'train', 'eval', 'test'};

load('../data/quality_sets_labeled_global_gallery_rr.mat');

src = '/media/guo/My Passport/ROOTDIR/FQDB/Frontals/4_train_eval_test_imgs/';

for k = 1:length(datatype_list)
    
    datatype = datatype_list{k};

    if strcmp(datatype, 'train')
        dst = '/media/guo/My Passport/ROOTDIR/FQDB/Frontals/5_trainset/original/';    
    elseif strcmp(datatype, 'eval')
        dst = '/media/guo/My Passport/ROOTDIR/FQDB/Frontals/6_evalset/original/';    
    else
        dst = '/media/guo/My Passport/ROOTDIR/FQDB/Frontals/7_testset/original/';    
    end    

    if ~exist([dst '/sseq/'], 'dir')
        mkdir([dst '/sseq/']);
    end
    if ~exist([dst '/bliinds/'], 'dir')
        mkdir([dst '/bliinds/']);
    end
    if ~exist([dst '/diivine/'], 'dir')
        mkdir([dst '/diivine/']);
    end
    if ~exist([dst '/brisque/'], 'dir')
        mkdir([dst '/brisque/']);
    end
    if ~exist([dst '/curvelet/'], 'dir')
        mkdir([dst '/curvelet/']);
    end
    if ~exist([dst '/niqe/'], 'dir')
        mkdir([dst '/niqe/']);
    end
    if ~exist([dst '/tmiqa/'], 'dir')
        mkdir([dst '/tmiqa/']);
    end

    for j = 1:length(labeled_sets)

        fprintf('%d/%d processing %s\n', j, length(labeled_sets), labeled_sets{j});

        if strcmp(datatype, 'train')
           src_path = [src labeled_sets{j} '/train/'];
        elseif strcmp(datatype, 'eval')
           src_path = [src labeled_sets{j} '/eval/'];
        else
           src_path = [src labeled_sets{j} '/test/'];
        end

        files = dir(fullfile(src_path, '*.jpg'));    

        k=1;

        sseqf = {}; bliindsf = {}; diivinef = {}; brisquef = {}; curveletf = {}; niqef = {}; tmiqa = {}; 

        for i = 1:length(files)

            I = imread([src_path '/' files(i).name]); 
            crgr = im2double(I);

            %% extract face recognition features
            sseqf{k,1} = labeled_sets{j,1};
            sseqf{k,2} = files(i).name;
            sseqf{k,3} = sseq_feature_extract(crgr,3);
            sseqf{k,4} = labeled_sets{j,3};

            bliindsf{k,1} = labeled_sets{j,1};
            bliindsf{k,2} = files(i).name;
            b = bliinds2_feature_extraction(crgr);
            bliindsf{k,3} = reshape(b,[1 24]);
            bliindsf{k,4} = labeled_sets{j,3};

            diivinef{k,1} = labeled_sets{j,1};
            diivinef{k,2} = files(i).name;
            diivinef{k,3} = divine_feature_extract(crgr); 
            diivinef{k,4} = labeled_sets{j,3};

            brisquef{k,1} = labeled_sets{j,1};
            brisquef{k,2} = files(i).name;
            brisquef{k,3} = brisque_feature(crgr);
            brisquef{k,4} = labeled_sets{j,3};

            curveletf{k,1} = labeled_sets{j,1};
            curveletf{k,2} = files(i).name;
            curveletf{k,3} = curvelet_features(imresize(crgr,[512 512]));
            curveletf{k,4} = labeled_sets{j,3};

            niqef{k,1} = labeled_sets{j,1};
            niqef{k,2} = files(i).name;
            niqef{k,3} = niqe_features(imresize(crgr,[128 128]));
            niqef{k,4} = labeled_sets{j,3};        

            tmiqaf{k,1} = labeled_sets{j,1};
            tmiqaf{k,2} = files(i).name;
            tmiqaf{k,3} = tmiqa_features(imresize(crgr,[128 128]));
            tmiqaf{k,4} = labeled_sets{j,3};        

            k=k+1;
        end

        data = sseqf; clear sseqf;
        save([dst '/sseq/' labeled_sets{j} ';sseq_features'], 'data', '-v7.3');

        data = bliindsf; clear bliindsf;
        save([dst '/bliinds/' labeled_sets{j} ';bliinds_features'], 'data', '-v7.3');

        data = diivinef; clear diivinef;
        save([dst '/diivine/' labeled_sets{j} ';diivine_features'], 'data', '-v7.3');

        data = brisquef; clear brisquef;
        save([dst '/brisque/' labeled_sets{j} ';brisque_features'], 'data', '-v7.3');

        data = curveletf; clear curveletf;   
        save([dst '/curvelet/' labeled_sets{j} ';curvelet_features'], 'data', '-v7.3');

        data = niqef; clear niqef;   
        save([dst '/niqe/' labeled_sets{j} ';niqe_features'], 'data', '-v7.3');

        data = tmiqaf; clear tmiqaf;   
        save([dst '/tmiqa/' labeled_sets{j} ';tmiqa_features'], 'data', '-v7.3');

    end
end
