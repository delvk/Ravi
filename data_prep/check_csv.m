function check_csv = check_csv(img_path,csv_path)
  pkg load image;
  dmap = csvread(csv_path);
  img = imread(img_path);
 imshow(img);
 figure()
  imagesc(dmap)
  
  endfunction