%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File to create grount truth density map for data set%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all;
pkg load image;
root = '../raw';
phase = 'train';
path = strcat(root, '/', phase,'_data');
img_path = strcat(path, '/images/');
gt_path = strcat(path, '/ground_truth/');
output_path = strcat(path,'/dmap/');
mkdir(output_path);
start = 1 
if (strcmp(phase,'train'))
  total = 400
elseif (strcmp(phase,'test'))
  total = 200
elseif (strcmp(phase,'val'))
  start = 201
  total = 316
else
  total = 0
endif
for i = start:total    
    if (mod(i,20)==0 || i == start || i == total)
        fprintf(1,'Processing %3d/%d files\n', i, total-start+1);
        fflush(1);
    end
    out_name = [output_path ,'DMAP_',num2str(i) '.csv'];
    if exist(out_name)
      t = csvread(out_name);
      if(size(t,1)!=768 || size(t,2)!=1024)
        fprintf(1,'%s height:%d, width: %d ==> Fixing\n', out_name, size(t,1), size(t,2))
      else
        continue
    endif
    load(strcat(gt_path, 'GT_IMG_',num2str(i),'.mat')) ;
    input_img_name = strcat(img_path,'IMG_',num2str(i),'.jpg');

    im = imread(input_img_name);
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end     
    annPoints =  image_info{1}.location;   
    [h, w, c] = size(im);
    im_density = get_density_map_gaussian(im,annPoints);    
    csvwrite(out_name, im_density);       
end
