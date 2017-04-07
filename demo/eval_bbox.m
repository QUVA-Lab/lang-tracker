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

signiture = 'results_lang_seg_sigmoid_thresh0.5';
d = dir(['/home/zhenyang/Workspace/devel/project/vision/text_objseg_caffe/results/OTB100/' signiture  '/']);
isub = [d(:).isdir]; %# returns logical vector
videos = {d(isub).name}';
videos(ismember(videos,{'.','..'})) = [];

counter = 1;
oas_all = {};
for vi = 1:numel(videos)
    video = videos{vi}
    f = dir(['/home/zhenyang/Workspace/devel/project/vision/text_objseg_caffe/results/OTB100/' signiture  '/' video '/*.jpg']);
    frames = {f(:).name}';
    frames(ismember(frames,{'.','..'})) = [];

    %frames = frames(1);
    pred_boxes = zeros(numel(frames), 4);
    rest_boxes = zeros(numel(frames), 4);
    oas = zeros(1, numel(frames));
    for fr = 1:numel(frames)
        im_name = frames{fr}(1:end-4);
        curHeatMapFile = ['/home/zhenyang/Workspace/devel/project/vision/text_objseg_caffe/results/OTB100/' signiture '/' video '/' im_name '.jpg'];
        curBBoxFile = ['/home/zhenyang/Workspace/devel/project/vision/text_objseg_caffe/results/OTB100/' signiture '/' video '/' im_name '.txt'];
        gtBBoxFile = ['/home/zhenyang/Workspace/data/OTB-100-othervideos/' video '/groundtruth_rect.txt'];

        boxData = dlmread(curBBoxFile);
        boxData_formulate = [boxData(1:4:end)' boxData(2:4:end)' boxData(1:4:end)'+boxData(3:4:end)' boxData(2:4:end)'+boxData(4:4:end)'];
        boxData_formulate = [min(boxData_formulate(:,1),boxData_formulate(:,3)),min(boxData_formulate(:,2),boxData_formulate(:,4)),max(boxData_formulate(:,1),boxData_formulate(:,3)),max(boxData_formulate(:,2),boxData_formulate(:,4))];

        gt_bboxes = dlmread(gtBBoxFile);
        gt_box = gt_bboxes(fr, 1:4);
        gt_box(3) = gt_box(1) + gt_box(3);
        gt_box(4) = gt_box(2) + gt_box(4);

        num_box = size(boxData_formulate, 1);
        if num_box > 1
            disp('There are more than 1 boxes generated!')
        end

        max_ov = 0;
        max_sz = 0;
        bebox = boxData_formulate(1, 1:4);
        for bb = 1:num_box
            pred_box = boxData_formulate(bb, 1:4);
            ov = IoU(pred_box, gt_box);
            
            box_sz = (pred_box(3) - pred_box(1))*(pred_box(4) - pred_box(2));
            if box_sz > max_sz
                max_ov = ov;
                max_sz = box_sz;
                bebox = pred_box;
            end
            %max_ov = max(ov, max_ov);
        end

        rest_boxes(fr, :) = bebox;
        pred_boxes(fr, :) = [bebox(1), bebox(2), bebox(3)-bebox(1), bebox(4)-bebox(2)];
        oas(1, fr) = max_ov;
    end

    predBBoxFile = ['/home/zhenyang/Workspace/devel/project/vision/text_objseg_caffe/results/OTB100/' signiture '/' video '_groundtruth_rect.txt'];
    dlmwrite(predBBoxFile, pred_boxes);

    restBBoxFile = ['/home/zhenyang/Workspace/devel/project/vision/text_obj_track/OTB100/lang_results/results_vgg16_lang_seg_fullconv/' video '_vgg16_lang_seg_fullconv.txt'];
    dlmwrite(restBBoxFile, rest_boxes);

    oas_all{counter} = oas;
    counter = counter + 1;
end

ovs = cat(2, oas_all{:});
size(ovs)
prec = mean(ovs)
recall = sum(ovs > 0.5) / numel(ovs)

