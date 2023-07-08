clear;clc;
% add path to external functions
addpath('../lib/redist-source');

% initial parameters
param.imageSize = [100 60];
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 6;
param.fc_prefilt = 4;
param.G = gistb;
[fb_real, fb_imag] = getGaborBank; 

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

    if ~exist([dst '/hog/'], 'dir')
        mkdir([dst '/hog/']);
    end
    if ~exist([dst '/gist/'], 'dir')
        mkdir([dst '/gist/']);
    end
    if ~exist([dst '/gabor/'], 'dir')
        mkdir([dst '/gabor/']);
    end
    if ~exist([dst '/lbp/'], 'dir')
        mkdir([dst '/lbp/']);
    end
    if ~exist([dst '/cnn/'], 'dir')
        mkdir([dst '/cnn/']);
    end

    for j = 1:length(labeled_sets)

        fprintf('%s set: %d/%d processing %s\n', datatype, j, length(labeled_sets), labeled_sets{j});

        if strcmp(datatype, 'train')
           src_path = [src labeled_sets{j} '/train/'];
        elseif strcmp(datatype, 'eval')
           src_path = [src labeled_sets{j} '/eval/'];
        else
           src_path = [src labeled_sets{j} '/test/'];
        end

        files = dir([src_path '/*.jpg']);

        k=1;

        hogf = {}; gistf = {}; gaborf = {}; lbpf = {}; cnnf = {};    

        for i = 1:length(files)

            I = imread([src_path '/' files(i).name]); 
            crgr = im2double(I);

            %% extract face recognition features
            hogf{k,1} = labeled_sets{j,1};
            hogf{k,2} = files(i).name;
            hogf{k,3} = hog(crgr);
            hogf{k,4} = labeled_sets{j,3};

            gistf{k,1} = labeled_sets{j,1};
            gistf{k,2} = files(i).name;
            gistf{k,3} = getGist(crgr, [], param);
            gistf{k,4} = labeled_sets{j,3};

            gaborf{k,1} = labeled_sets{j,1};
            gaborf{k,2} = files(i).name;
            gaborf{k,3} = gabor(crgr, fb_real, fb_imag);
            gaborf{k,4} = labeled_sets{j,3};

            lbpf{k,1} = labeled_sets{j,1};
            lbpf{k,2} = files(i).name;
            lbpf{k,3} = lbp(crgr);
            lbpf{k,4} = labeled_sets{j,3};

            cnnf{k,1} = labeled_sets{j,1};
            cnnf{k,2} = files(i).name;
            cnnf{k,3} = cnn(crgr);
            cnnf{k,4} = labeled_sets{j,3};

            k=k+1;
        end

        data = hogf; clear hogf;
        save([dst '/hog/' labeled_sets{j} ';hog_features'], 'data', '-v7.3');

        data = gistf; clear gistf;
        save([dst '/gist/' labeled_sets{j} ';gist_features'], 'data', '-v7.3');

        data = gaborf; clear gaborf;
        save([dst '/gabor/' labeled_sets{j} ';gabor_features'], 'data', '-v7.3');

        data = lbpf; clear lbpf;
        save([dst '/lbp/' labeled_sets{j} ';lbp_features'], 'data', '-v7.3');

        data = cnnf; clear cnnf;   
        save([dst '/cnn/' labeled_sets{j} ';cnn_features'], 'data', '-v7.3');
    end

end