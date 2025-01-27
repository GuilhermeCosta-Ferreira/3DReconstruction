%% Extraction Phase

% A window will open to add the folder, this will extract the folder data
clear;
folderPath = uigetdir('','Select the folder containing TIF files');
tifFiles = dir(fullfile(folderPath, '*.tif'));

imagesStore = cell(1, numel(tifFiles));
filteredStore = cell(1, numel(tifFiles));
level11Store = cell(1, numel(tifFiles));
level12Store = cell(1, numel(tifFiles));
level21Store = cell(1, numel(tifFiles));
level22Store = cell(1, numel(tifFiles));
otsuStore = cell(1, numel(tifFiles));
otsu1Store = cell(1, numel(tifFiles));
otsu2Store = cell(1, numel(tifFiles));
contourStore = cell(1, numel(tifFiles));
circled1Store = cell(1, numel(tifFiles));
circled2Store = cell(1, numel(tifFiles));
masked1Store = cell(1, numel(tifFiles));
masked2Store = cell(1, numel(tifFiles));

%This will iterate the files and save them to an array for easy access
for i = 1:numel(tifFiles)
    filePath = fullfile(folderPath, tifFiles(i).name);
    image = imread(filePath);
    imagesStore{i} = image;
end

%% Filtering

%Set-up
N = 1;
sigma = 1;
gaussianKernel = fspecial('gaussian', [N N], sigma);
medianKernel = ones(N)/(N^2);

%Applying
for i = 1:numel(tifFiles)
    filteredStore{i} = rescale(conv2(imagesStore{i},gaussianKernel, 'same'));
end

%{
imagem = imagesStore{11};
figure(10);
subplot(1, 4, 1);
imagesc(imagem);
colormap(gray(256)); % Definir colormap para escala de cinza
title('Original Image - Slice 11');
axis off;
pbaspect([1 1 1])
xlim([700, 1600]);
ylim([800, 1600]);

a = 2*(ceil(1.5))+1;
%imagem_filtrada = imgaussfilt(imagem,5,FilterSize=[a,a]);
imagem_filtrada = imgaussfilt(imagem,0.5);

subplot(1, 4, 2);
imagesc(imagem_filtrada);
colormap(gray(256)); % Definir colormap para escala de cinza
title('Filtered Image (0.5) - Slice 11');
axis off;
pbaspect([1 1 1])
xlim([700, 1600]);
ylim([800, 1600]);

b = 2*(ceil(3*1.5))+1;
%imagem_filtrada1 = imgaussfilt(imagem,5,FilterSize=[b,b]);
imagem_filtrada1 = imgaussfilt(imagem,1.5);

subplot(1, 4, 3);
imagesc(imagem_filtrada1);
colormap(gray(256)); % Definir colormap para escala de cinza
title('Filtered Image (1.5) - Slice 11');
axis off;
pbaspect([1 1 1])
xlim([700, 1600]);
ylim([800, 1600]);

c = 2*(ceil(50*1.5))+1;
%imagem_filtrada2 = imgaussfilt(imagem,5,FilterSize=[c,c]);
imagem_filtrada2 = imgaussfilt(imagem,3);

subplot(1, 4, 4);
imagesc(imagem_filtrada2);
colormap(gray(256)); % Definir colormap para escala de cinza
title('Filtered Image (3) - Slice 11');
axis off;
pbaspect([1 1 1])
xlim([700, 1600]);
ylim([800, 1600]);
%}


%% Segemntation - Density Ascend Algorithm
sRadius = 600;
eRadius = 100;
center1 = [1400, 1600];
center2 = [900, 600];
reducing_iterations = 2;
n_reductions = 20;
tolerance = 1;
tic;    % Starts counting search time
for i = 1:numel(filteredStore)
    image = filteredStore{i};
    
    % Finds the desired center of mass (density)
    [center1, figureFinal] = mean_shift(image, sRadius, eRadius, n_reductions, center1, 200, tolerance, reducing_iterations);
    maskedImage = apply_mask(image, eRadius, center1);
    circled1Store{i} = figureFinal;
    masked1Store{i} = maskedImage;

    [center2, figureFinal] = mean_shift(image, sRadius, eRadius, n_reductions, center2, 200, tolerance, reducing_iterations);
    maskedImage = apply_mask(image, eRadius, center2);
    circled2Store{i} = figureFinal;

    masked2Store{i} = maskedImage;
    if(reducing_iterations ~= 0)
        reducing_iterations = reducing_iterations - 1;
    end
end
elapsed_time_search = toc;
fprintf('Elapsed time for Search Algorithm: %.4f seconds\n', elapsed_time_search);

%figure(1);
%colormap(gray(256));
%imagesc(maskedImage);

%library_view(circled1Store,1);
%library_view(circled2Store,2);

%figure(3);
%colormap(gray(256));
%imagesc(maskedStore{11});


%% Segmentation - Otsu Algorithm

tic;    % Starts counting otsu time
%Threshold farming - 1 aggregate
for i = 1:numel(tifFiles)
    [level1, level2] = multithresh(rescale(masked1Store{i}),1);
    level11Store{i} = level1;
    level12Store{i} = level2;
end
level11Store = cell2mat(level11Store);
level12Store = cell2mat(level12Store);
levelAStore = movmean(level12Store, 10); %- ones(numel(level21Store))*0.2;
levelStores1 = [level11Store, 
    level12Store, 
    levelAStore];

%Otsu method - 1 aggregate
for i = 1:numel(tifFiles)
    otsu1Store{i} = imbinarize(rescale(masked1Store{i}), levelAStore(i));
end

%Threshold farming - 2 aggregate
for i = 1:numel(tifFiles)
    [level1, level2] = multithresh(rescale(masked2Store{i}),1);
    level21Store{i} = level1;
    level22Store{i} = level2;
end
level21Store = cell2mat(level21Store);
level22Store = cell2mat(level22Store);
levelAStore = movmean(level22Store, 10); %-ones(numel(level21Store))*0.2;
levelStores2 = [level21Store, 
    level22Store, 
    levelAStore];

%Otsu method - 2 aggregate
for i = 1:numel(tifFiles)
    otsu2Store{i} = imbinarize(rescale(masked2Store{i}), levelAStore(i));
end

%Sum all otsu
for i = 1:numel(tifFiles)
    otsuStore{i} = otsu1Store{i} + otsu2Store{i};
end

elapsed_time_otsu = toc;
fprintf('Elapsed time for Otsu Algorithm: %.4f seconds\n', elapsed_time_otsu);


%% Segmentation - Post-Processing E&D Algorithm
se = strel('disk',1);
se2 = strel('disk', 2);
aggregate_size = 0;
for i = 1:numel(otsuStore)
    %otsuStore{i} = imdilate(otsuStore{i}, se2);
    otsuStore{i} = imerode(otsuStore{i}, se);
    %otsuStore{i} = conv2(otsuStore{i}, gaussianKernel);
    otsuStore{i} = imdilate(otsuStore{i}, se2);
    aggregate_size = aggregate_size +  sum(otsuStore{i}(:));
end

fprintf('Aggregate Total: %d\n', aggregate_size);
%disp(se);

%% 3D Reconstruction

l = numel(otsuStore);
dim = size(otsuStore{1}, 1);
image3D = zeros(dim, dim, l);

for i = 1:l
    image3D(:,:,i) = otsuStore{i};
end

fprintf('image3D build\n');

figure(1);
clf(1);
p = patch(isosurface(double(image3D), 1e-4)); 
p.FaceColor = 'red';
p.EdgeColor = 'none';
axis tight; daspect([1,2,0.2]); view(3); camlight; lighting gouraud

fprintf('Finished\n');


%% 3D Alignement
%{
tic;
% Configura o otimizador e a métrica para o registro de imagens
[optimizer, metric] = imregconfig('monomodal');

% Define a 16ª imagem como a referência (fixa)
fixed = im2double(otsuStore{15});

% Alinha todas as imagens com a referência
for i = 1:numel(otsuStore)
    if i ~= 15 % Pula a 16ª imagem, já que é a referência
        moving = im2double(otsuStore{i});
        movingRegistered = imregister(moving, fixed, 'rigid', optimizer, metric);
        otsuStore{i} = movingRegistered; % Atualiza a imagem alinhada
    end
end

elapsed_time_alignement = toc;
fprintf('Elapsed time for Alignement: %.4f seconds\n', elapsed_time_alignement);

image3D = zeros(dim, dim, l);

for i = 1:l
    image3D(:,:,i) = otsuStore{i};
end

fprintf('image3D build\n');

figure(2);
clf(2);
p = patch(isosurface(double(image3D), 1e-4)); 
p.FaceColor = 'red';
p.EdgeColor = 'none';
axis tight; daspect([1,2,0.2]); view(3); camlight; lighting gouraud

fprintf('Finished\n');
%}


%% Visualization
%{
otsu_view(3, 1, imagesStore, filteredStore, otsuStore, levelStores)
otsu_view(4, 2, imagesStore, filteredStore, otsuStore, levelStores)

library_view(imagesStore,3)
library_view(otsuStore,4);
%}
%otsu_view(22, 2, circled2Store, masked2Store, otsuStore, levelStores2)
%otsu_view(8, 3, circled2Store, masked2Store, otsuStore, levelStores2)
%otsu_view(1, 4, circled2Store, masked2Store, otsuStore, levelStores1)
otsu_view(11, 6, circled1Store, masked1Store, otsuStore, levelStores1)
otsu_view(11, 7, circled2Store, masked2Store, otsuStore, levelStores2)
library_view(otsuStore,5);
%filtered_view(11,1,imagesStore, contourStore);

%threshold_view(9,1,filteredStore,levelStores, 0.48);

%{
figure(3);
colormap(gray(256));
imagesc(masked1Store{11});
pbaspect([1 1 1]);
%}


%% Saving Code


%Save Code
for i = 1:numel(otsuStore)
    savePath = fullfile('/Users/guilhermec.f/Documents/Faculdade/3º Ano/2º Semestre/FBIBI/Project/Save', sprintf('segmented_%d.png', i));
    imwrite(otsu2Store{i},savePath);
end
%}


%% Visualization Functions

%Function used to visualize a full library
function library_view(library, window)
    size =  numel(library);
    figure(window);
    colormap(gray(256));

    for i = 1:size
        subplot(5,6,i);
        imagesc(library{i});
        title(sprintf('Image Nº%d', i));
        pbaspect([1 1 1]);
    end
end

%Function used to compared filtered image with the original
function filtered_view(image, window, imagesStore, filteredStore)
    figure(window);
    colormap(gray(256));

    subplot(1,2,1);
    imagesc(imagesStore{image});
    title('Original Image');
    pbaspect([1 1 1]);

    subplot(1,2,2);
    imagesc(filteredStore{image});
    title('Filtered Image');
    pbaspect([1 1 1]);
end

%Function used to compare otsu segmentation with original and filtered images
function otsu_view(image, window, imagesStore, filteredStore, otsuStore, levelStores)
    figure(window);
    clf(window);
    colormap(gray(256));

    % Plot original image and histogram
    subplot(2,3,1);
    imagesc(imagesStore{image});
    title('Searched Image');
    subplot(2,3,4);
    imhist(imagesStore{image});
    title('Histogram');
    pbaspect([1 1 1]);

    % Plot filtered image and histogram
    subplot(2,3,2);
    imagesc(filteredStore{image});
    title('Image with Mask');
    subplot(2,3,5);
    imhist(filteredStore{image});
    title('Filtered Histogram');
    pbaspect([1 1 1]);

    % Plot Otsu thresholded image
    subplot(2,3,3);
    imagesc(otsuStore{image});
    title('Otsu Thresholded Image');
    pbaspect([1 1 1]);

    % Plot thresholds
    subplot(2,3,6);
    x = 1:numel(levelStores(1,:));
    plot(x, levelStores(1,:));
    hold on;
    plot(x, levelStores(2,:));
    hold on;
    plot(x, levelStores(3,:));
    title('Thresholds');
    pbaspect([1 1 1]);
    
    % Add legend for levelStores
    legend('Threshold 1', 'Threshold 2', 'Smoothed Threshold');
end

%Function used to study and chose better trhesolds in the otsu method
function [final_center, figure_final] = mean_shift(image, sRadius, eRadius, n_reductions, center, maximumIterations, tolerance, reduced_iterations)
    % Inputs:
    %       image - Input image data.
    %       radius - Radius of the circular mask used for mean shift.
    %       center - Initial center coordinates for mean shift.
    %       maximumIterations - Maximum number of iterations to run mean shift.
    %       sensitivity - Convergence threshold for mean shift algorithm.
    %
    % Outputs:
    %       final_center - Final estimated mode center after mean shift iterations.
    %       figure_final - Output image with circles representing mean shift iterations.

    centersStore{1} = zeros(1,2);
    centersStore{2} = center;

    size = numel(centersStore);
    i = 3;
    if(sRadius > eRadius)
        step = (sRadius-eRadius)/n_reductions;
    else
        step = 0;
    end
    radius = sRadius;

    if(reduced_iterations == 0)
        radius = eRadius;
        step = 0;
    end

    figure_final = insertShape(image, 'Circle', [centersStore{2}(1), centersStore{2}(2), radius], 'Color', 'cyan');
    while (~(norm(centersStore{size} - centersStore{size-1}) <= tolerance) && ~(i == maximumIterations)) || (radius > eRadius)
        [total_Intensity, locatedX_Intensity, locatedY_Intensity] = totalIntensity_masked(image, radius, centersStore{i-1});
        centersStore{i} = [locatedX_Intensity/total_Intensity, locatedY_Intensity/total_Intensity];
        figure_final = insertShape(figure_final, 'Circle', [centersStore{i}(1), centersStore{i}(2), radius], 'Color', 'cyan');
        i = i + 1;
        size = numel(centersStore);
        if(radius > eRadius) 
            radius = radius - step;
        end
    end
    figure_final = insertShape(figure_final, 'Circle', [centersStore{size}(1), centersStore{size}(2), radius], 'Color', 'red');

    fprintf('Early Stop? %s\n', string(i == maximumIterations));
    fprintf('Times Runned: %d\n', i);
    fprintf('Radius Final Size: %d\n', radius);
    fprintf('Step Size: %d\n', step);
    fprintf('\n');
    final_center = centersStore{size};
end

%Computes the total intensity, total intensity weighted by x-coordinate and then y-coordinate, within a circular region in the image.
function [total_Intensity, locatedX_Intensity, locatedY_Intensity] = totalIntensity_masked(image, radius, center)
    % Inputs:
    %       image: A grayscale image matrix.
    %       radius: The radius of the circular region.
    %       center: The center coordinates (x, y) of the circular region.
    %
    % Outputs:
    %       total_Intensity: The total intensity within the circular region.
    %       locatedX_Intensity: The sum of x-coordinate times intensity within the circular region.
    %       locatedY_Intensity: The sum of y-coordinate times intensity within the circular region.

    % Setup
    [rows, cols] = size(image);
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    % Gets the masked image (inside the circle of radius bandwith and center location)
    distances = sqrt((X - center(1)).^2 + (Y - center(2)).^2);
    pointsWithinCircle = distances <= radius;
    maskedImage = image(pointsWithinCircle);
    
    % Calculate total intensity
    total_Intensity = sum(maskedImage);
    
    % Calculate x and y coordinates within the circle
    x_coords = X(pointsWithinCircle);
    y_coords = Y(pointsWithinCircle);
    
    % Calculate the product of x-coordinate times intensity
    locatedX_Intensity = sum(x_coords .* double(maskedImage));
    
    % Calculate the product of y-coordinate times intensity
    locatedY_Intensity = sum(y_coords .* double(maskedImage));
end

%Applies the mask and eliminates everything outside
function maskedImage = apply_mask(image, radius, center)
    [rows, cols] = size(image);
    [X, Y] = meshgrid(1:cols, 1:rows);
    distances = sqrt((X - center(1)).^2 + (Y - center(2)).^2);
    mask = distances <= radius;
    maskedImage = double(image) .* mask;
end