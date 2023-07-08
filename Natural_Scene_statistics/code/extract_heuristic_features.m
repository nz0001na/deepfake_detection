clear;clc;
% add path to external functions
addpath('../lib/heuristic_features/face-release1.0-basic');
addpath('../lib/heuristic_features/');
addpath('../common/');

datatype_list = {'train', 'eval', 'test'};

load('../data/quality_sets_labeled_global_gallery_rr.mat');

src = '/media/guo/My Passport/ROOTDIR/FQDB/Frontals/4_train_eval_test_imgs/';

base_path = '/media/guo/My Passport/ROOTDIR/FQDB/Frontals/0_originals';

for k = 2:length(datatype_list)
    
    datatype = datatype_list{k};
    
    if strcmp(datatype, 'train')
        dst = '/media/guo/My Passport/ROOTDIR/FQDB/Frontals/5_trainset/original/';    
    elseif strcmp(datatype, 'eval')
        dst = '/media/guo/My Passport/ROOTDIR/FQDB/Frontals/6_evalset/original/';    
    else
        dst = '/media/guo/My Passport/ROOTDIR/FQDB/Frontals/7_testset/original/';    
    end

    if ~exist([dst '/heuristic/'], 'dir')
        mkdir([dst '/heuristic/']);
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

        hf = {}; 

        for i = 1:length(files)

            I = imread([src_path '/' files(i).name]); 
%             crgr = im2double(I);

            %% extract face recognition features
            hf{k,1} = labeled_sets{j,1};
            hf{k,2} = files(i).name;
            parts1 = strsplit(files(i).name, '.');
            if length(parts1) > 2
                parts2 = strjoin(parts1(1:end-1),'.');
                parts3 = strsplit(parts2,'_');
                file_name = [strjoin(parts3(1:end-1),'_') '.' parts1{end}];                
            else
                parts2 = strsplit(parts1{1:end-1},'_');
                file_name = [strjoin(parts2(1:end-1),'_') '.' parts1{end}];
            end
            file_path = get_original_path_from_qualityset(labeled_sets{j,1});
            hf{k,3} = heuristic_features(I, fullfile(base_path, file_path, file_name));
            hf{k,4} = labeled_sets{j,3};

            k=k+1;
        end

        data = hf; clear hf;
        save([dst '/heuristic/' labeled_sets{j} ';heuristic_features'], 'data', '-v7.3');
    end

end