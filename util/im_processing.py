from __future__ import absolute_import, division, print_function

import skimage.transform
import numpy as np

def resize_and_pad(im, input_h, input_w):
    # Resize and pad im to input_h x input_w size
    im_h, im_w = im.shape[:2]
    scale = min(input_h / im_h, input_w / im_w)
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))
    pad_h = int(np.floor(input_h - resized_h) / 2)
    pad_w = int(np.floor(input_w - resized_w) / 2)

    resized_im = skimage.transform.resize(im, [resized_h, resized_w])
    if im.ndim > 2:
        new_im = np.zeros((input_h, input_w, im.shape[2]), dtype=resized_im.dtype)
    else:
        new_im = np.zeros((input_h, input_w), dtype=resized_im.dtype)
    new_im[pad_h:pad_h+resized_h, pad_w:pad_w+resized_w, ...] = resized_im

    return new_im

def resize_and_crop(im, input_h, input_w):
    # Resize and crop im to input_h x input_w size
    im_h, im_w = im.shape[:2]
    scale = max(input_h / im_h, input_w / im_w)
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))
    crop_h = int(np.floor(resized_h - input_h) / 2)
    crop_w = int(np.floor(resized_w - input_w) / 2)

    resized_im = skimage.transform.resize(im, [resized_h, resized_w])
    if im.ndim > 2:
        new_im = np.zeros((input_h, input_w, im.shape[2]), dtype=resized_im.dtype)
    else:
        new_im = np.zeros((input_h, input_w), dtype=resized_im.dtype)
    new_im[...] = resized_im[crop_h:crop_h+input_h, crop_w:crop_w+input_w, ...]

    return new_im

def crop_and_pad_bboxes_subtract_mean(im, bboxes, crop_size, image_mean):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))

    im_h, im_w = im.shape[:2]
    num_bbox = bboxes.shape[0]
    imcrop_batch = np.zeros((num_bbox, crop_size, crop_size, 3), dtype=np.float32)

    # if need padding from largest sclaed box
    offset_x1 = 0
    offset_y1 = 0
    box = bboxes[-1, :]
    if box[0] < 1 or box[1] < 1 or box[2] > im_w or box[3] > im_h:
        offset_x1 = np.maximum(1 - box[0], 0)
        offset_y1 = np.maximum(1 - box[1], 0)          
        offset_x2 = np.maximum(box[2] - im_w, 0)
        offset_y2 = np.maximum(box[3] - im_h, 0)
        im_pad = np.zeros((im_h+offset_y1+offset_y2,im_w+offset_x1+offset_x2,3), 'uint8')
        im_pad[offset_y1:im_h+offset_y1, offset_x1:im_w+offset_x1, :] = im
        im = im_pad

    im = skimage.img_as_ubyte(im)
    for n_bbox in range(bboxes.shape[0]):
        xmin, ymin, xmax, ymax = bboxes[n_bbox]
        # crop and resize
        # imcrop = im[ymin:ymax+1, xmin:xmax+1, :]
        imcrop = im[ymin+offset_y1-1:ymax+offset_y1, xmin+offset_x1-1:xmax+offset_x1, :]
        #print('UserWarning: Possible precision loss when converting from float64 to uint8')
        imcrop_resize = skimage.img_as_ubyte(
                        skimage.transform.resize(imcrop, [crop_size, crop_size]))
        imcrop_batch[n_bbox, ...] = imcrop_resize - image_mean
    #imcrop_batch -= image_mean
    return imcrop_batch

def crop_bboxes_subtract_mean(im, bboxes, crop_size, image_mean):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))

    im = skimage.img_as_ubyte(im)
    num_bbox = bboxes.shape[0]
    imcrop_batch = np.zeros((num_bbox, crop_size, crop_size, 3), dtype=np.float32)
    for n_bbox in range(bboxes.shape[0]):
        xmin, ymin, xmax, ymax = bboxes[n_bbox]
        # crop and resize
        imcrop = im[ymin:ymax+1, xmin:xmax+1, :]
        #print('UserWarning: Possible precision loss when converting from float64 to uint8')
        imcrop_resize = skimage.img_as_ubyte(
                        skimage.transform.resize(imcrop, [crop_size, crop_size]))
        imcrop_batch[n_bbox, ...] = imcrop_resize - image_mean
    #imcrop_batch -= image_mean
    return imcrop_batch

def bboxes_from_masks(masks):
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    num_mask = masks.shape[0]
    bboxes = np.zeros((num_mask, 4), dtype=np.int32)
    for n_mask in range(num_mask):
        idx = np.nonzero(masks[n_mask])
        xmin, xmax = np.min(idx[1]), np.max(idx[1])
        ymin, ymax = np.min(idx[0]), np.max(idx[0])
        bboxes[n_mask, :] = [xmin, ymin, xmax, ymax]
    return bboxes

def crop_masks_subtract_mean(im, masks, crop_size, image_mean):
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    num_mask = masks.shape[0]

    im = skimage.img_as_ubyte(im)
    bboxes = bboxes_from_masks(masks)
    imcrop_batch = np.zeros((num_mask, crop_size, crop_size, 3), dtype=np.float32)
    for n_mask in range(num_mask):
        xmin, ymin, xmax, ymax = bboxes[n_mask]

        # crop and resize
        im_masked = im.copy()
        mask = masks[n_mask, ..., np.newaxis]
        im_masked *= mask
        im_masked += image_mean.astype(np.uint8) * (1 - mask)
        imcrop = im_masked[ymin:ymax+1, xmin:xmax+1, :]
        imcrop_batch[n_mask, ...] = skimage.img_as_ubyte(skimage.transform.resize(imcrop, [224, 224]))

    imcrop_batch -= image_mean
    return imcrop_batch

def crop_featmap_center(im):
    im_h, im_w = im.shape[1:]
    crop_h = int(np.floor(im_h) / 2)
    crop_w = int(np.floor(im_w) / 2)
    xmin, ymin, xmax, ymax = np.round([0.5*crop_w+1, 0.5*crop_h+1, 1.5*crop_w, 1.5*crop_h]).astype(int)
    # crop feature map
    imcrop = np.copy(im[:, ymin-1:ymax, xmin-1:xmax])
    return imcrop

def crop_featmap_from_center(im, ratio):
    im_h, im_w = im.shape[1:]
    crop_h = int(np.floor(im_h) / ratio)
    crop_w = int(np.floor(im_w) / ratio)
    xmin, ymin, xmax, ymax = np.round([0.5*(ratio-1)*crop_w+1, 0.5*(ratio-1)*crop_h+1, 0.5*(ratio+1)*crop_w, 0.5*(ratio+1)*crop_h]).astype(int)
    # crop feature map
    imcrop = np.copy(im[:, ymin-1:ymax, xmin-1:xmax])
    return imcrop
