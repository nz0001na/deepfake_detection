
img = 'si2104_0004.jpg'
img = '2.jpg'
a = exist(img)

f = SRM(img)
fields = fieldnames(f);
len = size(fields,1)

final_feature = []
for m=1:len
    field = fields{m}
    value = getfield(f,field)
    final_feature = [final_feature,value]
%     print 'd'
    
end


% print 'd'
