function [gt_targets_IoG,label_IoG,label_Box] = proposal_compute_targets_IoG(conf, gt_rois, gt_ignores, gt_labels, ex_rois, image_roidb, im_scale)
    
    % output: gt_targets_IoG
    %   positive: [class_label, gt_targets]
    gt_rois = gt_rois(gt_ignores~=1, :);

    if(isempty(gt_rois))
        gt_targets_IoG = zeros(size(ex_rois));
        label_IoG = zeros(size(ex_rois,1),1);
        label_Box = zeros(size(ex_rois,1),1);
        return;
    end
    
    % drop anchors which run out off image boundaries, if necessary
    contained_in_image = is_contain_in_image(ex_rois, round(image_roidb.im_size * im_scale));
    contained_in_image_ind = find(contained_in_image);
   
    
    % added by zzf
    ex_gt_overlaps_IoG = boxoverlap(ex_rois, gt_rois); % for fg
    
    [overlaps_IoG,targets_IoG] = max(ex_gt_overlaps_IoG,[],2);
    label_Box = targets_IoG;
    no_intersect = find(overlaps_IoG==0);
    label_Box(no_intersect,:)=0;
    
    rows_IoG = 1:size(targets_IoG,1);
    rows_IoG = rows_IoG';
    ex_gt_overlaps_IoG(sub2ind(size(ex_gt_overlaps_IoG),rows_IoG,targets_IoG))=0;
    [ex_max_overlaps_IoG,targets_IoG] = max(ex_gt_overlaps_IoG,[],2);
    fg_inds_IoG = unique([find(ex_max_overlaps_IoG >= conf.fg_thresh)]);
    fg_inds_IoG = intersect(fg_inds_IoG, contained_in_image_ind);
    
    bg_inds_IoG = 1:size(ex_gt_overlaps_IoG,1);
    bg_inds_IoG = bg_inds_IoG';
    bg_inds_IoG = setdiff(bg_inds_IoG,fg_inds_IoG); 
    
    %targets_IoG(bg_inds_IoG,:)=0;
    
    gt_targets_IoG = gt_rois(targets_IoG,:);
    label_IoG = ones(size(ex_rois,1),1);
    
    gt_targets_IoG(bg_inds_IoG,:) = 0;
    label_IoG(bg_inds_IoG,:) = 0; 
    
    
end

function contained = is_contain_in_image(boxes, im_size)
    contained = boxes >= 1 & bsxfun(@le, boxes, [im_size(2), im_size(1), im_size(2), im_size(1)]);
    
    contained = all(contained, 2);
end
