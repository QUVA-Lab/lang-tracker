%% Here is the code to generate the bounding box from the heatmap
%
% to reproduce the ILSVRC localization result, you need to first generate
% the heatmap for each testing image by merging the heatmap from the
% 10-crops (it is exactly what the demo code is doing), then resize the merged heatmap back to the original size of
% that image. Then use this bbox generator to generate the bbox from the resized heatmap.
%
% The source code of the bbox generator is also released. Probably you need
% to install the correct version of OpenCV to compile it.
%
% Special thanks to Hui Li for helping on this code.
%
% Bolei Zhou, April 19, 2016

bbox_threshold = [20, 100, 110]; % parameters for the bbox generator
curParaThreshold = [num2str(bbox_threshold(1)) ' ' num2str(bbox_threshold(2)) ' ' num2str(bbox_threshold(3))];

d = dir('../results/OTB100/results_lang_seg_sigmoid_thresh0.5/');
isub = [d(:).isdir]; %# returns logical vector
videos = {d(isub).name}';
videos(ismember(videos,{'.','..'})) = [];

for vi = 1:numel(videos)
    video = videos{vi}
    f = dir(['../results/OTB100/results_lang_seg_sigmoid_thresh0.5/' video '/*.jpg']);
    frames = {f(:).name}';
    frames(ismember(frames,{'.','..'})) = [];

    for fr = 1:numel(frames)
        im_name = frames{fr}(1:end-4);
        curHeatMapFile = ['../results/OTB100/results_lang_seg_sigmoid_thresh0.5/' video '/' im_name '.jpg'];
        curImgFile = ['/home/zhenyang/Workspace/data/OTB-100-othervideos/' video '/img/' im_name '.jpg'];
        curBBoxFile = ['../results/OTB100/results_lang_seg_sigmoid_thresh0.5/' video '/' im_name '.txt'];
        system(['bboxgenerator/./dt_box ' curHeatMapFile ' ' curParaThreshold ' ' curBBoxFile]);
    end

end
