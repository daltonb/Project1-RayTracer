%% Load hi-res texture images
w1_full = imread('wrinkled1_full.JPG');
w_full = imread('wrinkled2_full.JPG');
w3_full = imread('wrinkled3_full.JPG');
w4_full = imread('wrinkled4_full.JPG');
w5_full = imread('wrinkled5_full.JPG');

%% Crop and resize textures
w1=w1_full(:,4752-3168-1:end,:);
w1 = imresize(w1, [600 600]);
w2 = imresize(w2_full, [600 900]);
w3 = imresize(w3_full, [600 900]);
w4 = imresize(w4_full, [600 900]);
w5 = imresize(w5_full, [600 900]);

%% Write images
imwrite(w1, 'wrinkled_back_face.ppm', 'Encoding', 'ASCII')
imwrite(w2, 'wrinkled_left_face.ppm', 'Encoding', 'ASCII')
imwrite(w3, 'wrinkled_right_face.ppm', 'Encoding', 'ASCII')
imwrite(w4, 'wrinkled_top_face.ppm', 'Encoding', 'ASCII')
imwrite(w5, 'wrinkled_bottom_face.ppm', 'Encoding', 'ASCII')