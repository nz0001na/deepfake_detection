function f = call_feature_function_ffhq_zn(impath, n, reso)

I = imread(impath); 
img = im2double(I);
crgr = imresize(img,[reso reso])


crgr_gray = rgb2gray(crgr);   

%% extract NNS features

%   (1) SSEQ
if n == 1
    f = sseq_feature_extract(crgr,3);
end

%   (2) BRISQUE
if n==2
    f = brisque_feature(crgr_gray);
end

%   *(3) BLIINDS
if n==6
    b = bliinds2_feature_extraction(crgr);
    f = reshape(b,[1 24]);
end

%   *(4) DIIVINE
if n==7
    f = divine_feature_extract(crgr); 
end

%   *(5) CURVELET
if n==3
    f = curvelet_features(imresize(crgr,[512 512]));
end

%   *(6) NIQE 
if n==4
    f = niqe_features(imresize(crgr,[128 128]));
end

% %    *(7) TIMIQA
if n==5
    f = tmiqa_features(imresize(crgr_gray,[128 128]));    
end

