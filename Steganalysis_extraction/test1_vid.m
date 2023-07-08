clear;clc;

src_path = 'C:/000_NaZhangBack/2_deepfake/steg_code/vid_crop_face/'
% src_path = 'C:\000_NaZhangBack\1_deepfake\00_data\vid_timit_crop_dlib\'
dst_path = 'C:\000_NaZhangBack\2_deepfake\00_data\vid_timit_crop_dlib\'
if ~exist(dst_path, 'dir')
    mkdir(dst_path)
end
    
sub_item = dir(src_path)
sub_item(ismember( {sub_item.name}, {'.', '..'})) = [];
%     'VidTIMIT','lower_quality','higher_quality')
for m=1:length(sub_item)
    sub = sub_item(m).name
    if strcmp(sub, 'VidTIMIT')==0
        label=0
    else
        label=1
    end
    id_list = dir([src_path sub '\'])
    id_list(ismember({id_list.name},{'.','..'})) = [];
    for p=1:length(id_list)
        id = id_list(p).name
        dst_p = [dst_path sub '\' id '\']
        if ~exist(dst_p, 'dir')
            mkdir(dst_p)
        end
        sne_item = dir([src_path sub '\' id '\'])
        sne_item(ismember({sne_item.name},{'.','..'}))=[]
        for q=1:length(sne_item)
            sne_name = sne_item(q).name
            img_list = dir([src_path sub '\' id '\' sne_name '\'])
            img_list(ismember({img_list.name},{'.','..'}))=[]
            features = []
            labels = []
            for r=1:length(img_list)
                name = img_list(r).name
                img = [src_path sub '\' id '\' sne_name '\' name];
                f = SRM(img)
                fields = fieldnames(f);
                len1 = size(fields,1)
                final_feature = []
                for m=1:len1
                       field = fields{m}
                       value = getfield(f,field)
                       final_feature = [final_feature,value]
                end


                features = [features; final_feature]
                labels = [labels;label]           
            end   

            save([dst_p sne_name '.mat'],'features','labels')

        end

    end
end






