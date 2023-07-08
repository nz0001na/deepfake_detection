

img_path = '/media/guo/500B/1_code/steg_code/crop_face/'
folder_list = dir(img_path)
folder_list(ismember( {folder_list.name}, {'.', '..'})) = [];  
len = size(folder_list,1)
for i=1:len
    fold_name = folder_list(i).name
    
    fold_path = [img_path fold_name '/'];
    img_list = dir(fold_path)
    img_list(ismember({img_list.name},{'.','..'})) = [];
    count = size(img_list, 1)
    for j=1:count
        img_name = img_list(j).name;
        img_file = [fold_path img_name]
        f = SRM(img_file)
        
        fields = fieldnames(f);
        len = size(fields,1)

        final_feature = []
        for m=1:len
            field = fields{m}
            value = getfield(f,field)
            final_feature = [final_feature,value]
            
        end
    end
end



print 'd'