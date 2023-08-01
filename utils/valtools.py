import os
import glob
import pathlib

import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from .metrics import box_iou
from .plots import save_one_box


def filtered_tpfpfn(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        tp (array[N, 10]), for 10 IoU levels
        fp (array[N, 10]), for 10 IoU levels
        fn (array[N, 10]), for 10 IoU levels
    """
    TP = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    FP_cls = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    fn_idx = np.array([])
    
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    
    for i in range(len(iouv)):
        # condi -> len(condi): 2, condi[0]: row(x) point, condi[1]: columns(y) point
        condi_iou = torch.where((iou >= iouv[i]))  # IoU > threshold
        condi_tp = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        
        # TP
        if condi_tp[0].shape[0]: 
            matches_tp = torch.cat((torch.stack(condi_tp, 1), iou[condi_tp[0], condi_tp[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if condi_tp[0].shape[0] > 1:
                # max iou
                matches_tp = matches_tp[matches_tp[:, 2].argsort()[::-1]] # sort iou
                # drop duplicate - detect
                matches_tp = matches_tp[np.unique(matches_tp[:, 1], return_index=True)[1]] # [0]:unique, [1]:indices
                # matches = matches[matches[:, 2].argsort()[::-1]]
                
                # drop duplicate - label
                matches_tp = matches_tp[np.unique(matches_tp[:, 0], return_index=True)[1]]
            # print('iouv:', iouv[i], 'matches tp:', matches_tp)
            TP[matches_tp[:, 1].astype(int), i] = True
        
        # FP_cls
        if condi_iou[0].shape[0]:
            matches = torch.cat((torch.stack(condi_iou, 1), iou[condi_iou[0], condi_iou[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if condi_iou[0].shape[0] > 1:
                # max iou
                matches = matches[matches[:, 2].argsort()[::-1]] # sort iou
                # drop duplicate - detect
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]] # [0]:unique, [1]:indices
                # matches = matches[matches[:, 2].argsort()[::-1]]

                # drop duplicate - label
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            FP_cls[matches[:, 1].astype(int), i] = True
            if 'matches_tp' in locals():
                FP_cls[matches_tp[:, 1].astype(int), i] = False
        
        # FN, FP, IoU@.5
        if i == 0:
            condi_iou_x = torch.where((iou < iouv[i]))  # IoU < threshold(0.5)
            matches_fn = torch.cat((torch.stack(condi_iou_x, 1), iou[condi_iou_x[0], condi_iou_x[1]][:, None]), 1).cpu().numpy()
            matches_fn = matches_fn[np.unique(matches_fn[:, 0], return_index=True)[1]]
            matches_fp = matches_fn[np.unique(matches_fn[:, 1], return_index=True)[1]]

            if 'matches_tp' in locals():
                fn_idx = np.setdiff1d(matches_fn[:, 0], matches_tp[:, 0]) # fn label - tp label
                fp_idx = np.setdiff1d(matches_fp[:, 1], matches_tp[:, 1]) # fp detect - tp detect 

            elif 'matches_tp' not in locals():
                fn_idx = matches_fn[:, 0]
                fp_idx = matches_fp[:, 1]

            fp_cls_idx = np.where(FP_cls[:, 0]==True)[0]
            if fp_cls_idx.size > 0:
                fp_idx = np.setdiff1d(fp_idx, fp_cls_idx) 
            
    TP = torch.tensor(TP, dtype=torch.bool, device=iouv.device)
    FP_cls = torch.tensor(FP_cls, dtype=torch.bool, device=iouv.device)
    return TP, FP_cls, fp_idx, fn_idx


def plot_val_tools(im, path, dst_path, anns, matrix, names, tpfpfn_per_cls, crop_save=False, img_save=False):
    from utils.plots import Colors
    colors = Colors()
    im_cp = (im[0]*255).clone().cpu().numpy()
    im_cp = np.transpose(im_cp, (1,2,0)).astype(np.uint8)
    im_cp = im_cp.copy()
    im_crop = im_cp.copy()
#     im_cp = cv2.cvtColor(im_cp, cv2.COLOR_BGR2RGB).copy()

    *_, h, w = im_cp.shape
    name_idx_dict = {val:key for key, val in names.items()}
    
    if not len(anns):
        return tpfpfn_per_cls, None 
    else:
        for ann in anns:
            # from target
            if matrix == 'FN':
                pt1 = ann[1:3].cpu().numpy().astype(int)
                pt2 = ann[3:].cpu().numpy().astype(int)
                class_name = names.get(ann[0].item())
                conf = None
                xyxy = ann[1:].cpu().numpy()
            # from pred
            else:
                pt1 = ann[:2].cpu().numpy().astype(int)
                pt2 = ann[2:4].cpu().numpy().astype(int)
                class_name = names.get(ann[-1].item())
                conf = round(ann[4].item(), 2)
                xyxy = ann[:4].cpu().numpy()
            text = class_name+' '+ str(conf) if conf is not None else class_name
            tpfpfn_per_cls[class_name][matrix] += 1
            cv2.rectangle(im_cp, pt1, pt2, colors(name_idx_dict[class_name]), 2)
            text_box_size, text_cnt = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 0.7, 1)
            cv2.rectangle(im_cp, pt1, [pt1[0]+text_box_size[0], pt1[1]-text_box_size[1]], 
                          colors(name_idx_dict[class_name]), -1)
            cv2.putText(im_cp, text, pt1, cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(im_cp, path.stem, (7, 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            matrix_box_size, _ = cv2.getTextSize(matrix, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            cv2.putText(im_cp, matrix, [h-matrix_box_size[0], w+matrix_box_size[1]], 
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            if crop_save:
                (dst_path/'crop'/matrix).mkdir(parents=True, exist_ok=True)
                save_one_box(xyxy, im_crop, file=dst_path / 'crop' / matrix / class_name / f'{path.stem}.jpg', BGR=False)
                # save_thumbnail(dst_path / matrix / class_name)
            if img_save:
                try : 
                    (dst_path/matrix).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(dst_path / matrix / f'{path.stem}_{matrix}.jpg', im_cp)
                except Exception as e:
                    print(e, 'no!!!!!!!!!')
                    raise
        # if crop_save:
            # save_thumbnail(dst_path / matrix)

        return tpfpfn_per_cls, im_cp


def merge_viz_imgs(imgs:list, path, dst_path):
    matrix = ['TP', 'FP_cls', 'FP', 'FN']
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    axes = ax.ravel()
    for idx, (mat, img) in enumerate(zip(matrix, imgs)):
        if img is None:
            continue
        else:
            axes[idx].imshow(img)
            axes[idx].set_title(mat)
    [axis.set_axis_off() for axis in axes]
    plt.tight_layout()
#     dst_path_v2 = dst_path / 'merge' / f'{path.stem}_vt.jpg'
    dst_path_v2 = dst_path / f'{path.stem}_vt.jpg'
    (dst_path_v2.parent).mkdir(parents=True, exist_ok=True)
#     os.makedirs(os.path.dirname(dst_path_v2), exist_ok=True)
    plt.savefig(dst_path_v2, dpi=150, 
                facecolor='#FFFFFF', bbox_inches='tight')
    plt.close()
    
    
def save_thumbnail(target_dir):
    '''
    description: crop 이미지들의 썸네일 생성
    input
        - target_dir : crop 이미지가 있는 경로
            - type : str
            - e.g. : 'val/exp/val_tools/FP'
    output: None
    '''
    font = cv2.FONT_HERSHEY_PLAIN
    
    if not os.path.isdir(target_dir):
        return None
    if isinstance(target_dir, str):
        target_dir = target_dir[:-1] if target_dir[-1] == '/' else target_dir
    class_names = [val for val in os.listdir(target_dir) 
                    if os.path.isdir(os.path.join(target_dir, val))]
    matrix = os.path.basename(target_dir)
    for c_name in tqdm(class_names, desc=f'make {matrix} thumbnail...'):
        final_w, final_h = 0, 0
        concat_img_dict = {}
        width_ls = []
        height_ls = []
        target_paths = glob.glob(os.path.join(target_dir, c_name, '*[!0-9].jpg'))
        if not len(target_paths):
            target_paths = glob.glob(os.path.join(target_dir, c_name, '*.jpg'))
            filename_len = min(map(lambda x: len(pathlib.Path(x).stem), target_paths))
            target_paths = [path for path in target_paths if len(pathlib.Path(path).stem)==filename_len]
        tmp_text_size, _ = cv2.getTextSize(os.path.basename(target_paths[0]), font, 1, 1)
        for target_path in target_paths:
            img_path_ls = glob.glob(os.path.splitext(target_path)[0]+'*[0-9].jpg')
            img_path_ls.append(target_path)
            img_shape_ls = np.array(list(map(lambda x: cv2.imread(x).shape, img_path_ls)))
            height_idx = img_shape_ls[:, 0].argsort()[::-1]
            h = img_shape_ls[height_idx][0, 0]
            height_ls.append(h)
            w_sum = img_shape_ls[:, 1].sum()
            width_ls.append(w_sum)
            bg = np.ones((h, w_sum, 3), dtype=np.uint8)*255
            target_w = 0
            for idx in height_idx:
                img = cv2.imread(img_path_ls[idx])
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_h, img_w, _ = img.shape
                bg[0:img_h, target_w:target_w+img_w] = img
                target_w += img_w
            concat_img_dict[os.path.basename(target_path)] = bg
        final_h, final_w = sum(height_ls), max(width_ls)
        # img filename text size : 330, 10
        if final_w < tmp_text_size[0]:
            final_w += tmp_text_size[0]
        if final_h < tmp_text_size[1]:
            final_h += tmp_text_size[1]
        final = np.ones((final_h, final_w, 3), dtype=np.uint8)*255
        target_h = 0
        for filename, concat_img in concat_img_dict.items():
            cimg_h, cimg_w, _ = concat_img.shape
            final[target_h:target_h+cimg_h, 0:cimg_w] = concat_img
            target_h += cimg_h
            text_size, _ = cv2.getTextSize(filename, font, 1, 1)
            cv2.rectangle(final, [final_w-text_size[0], target_h-text_size[1]-2], [final_w, target_h], (255,255,255), -1)
            cv2.putText(final, filename, (final_w-text_size[0], target_h-2), font, 1, (0,0,0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(target_dir, f'{c_name}_{matrix}.jpg'), final)
        # plt.imshow(final)


def ext_each_metrics(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        tp (array[N, 10]), for 10 IoU levels
        fp (array[N, 10]), for 10 IoU levels
        fn (array[N, 10]), for 10 IoU levels
    """
    TP = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    FP_cls = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    fn_idx = np.array([])
    
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    not_correct_class = labels[:, 0:1] != detections[:, 5]

    # IoU threshold 0.5
    i = 0
    # condi -> len(condi): 2, condi[0]: row(x) point, condi[1]: columns(y) point
    condi_iou = torch.where((iou >= iouv[i]))  # IoU > threshold
    condi_tp = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
    condi_fp = torch.where((iou >= iouv[i]) & not_correct_class)
    condi_iou_x = torch.where((iou < iouv[i]))  # IoU < threshold
    # TP
    if condi_tp[0].shape[0]: 
        matches_tp = torch.cat((torch.stack(condi_tp, 1), iou[condi_tp[0], condi_tp[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
        if condi_tp[0].shape[0] > 1:
            # max iou
            matches_tp = matches_tp[matches_tp[:, 2].argsort()[::-1]] # sort iou
            # drop duplicate - detect
            matches_tp = matches_tp[np.unique(matches_tp[:, 1], return_index=True)[1]] # [0]:unique, [1]:indices
            # drop duplicate - label
            matches_tp = matches_tp[np.unique(matches_tp[:, 0], return_index=True)[1]]
        TP[matches_tp[:, 1].astype(int), i] = True

    # FP_cls
    if condi_iou[0].shape[0]:
        matches = torch.cat((torch.stack(condi_iou, 1), iou[condi_iou[0], condi_iou[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
        if condi_iou[0].shape[0] > 1:
            # max iou
            matches = matches[matches[:, 2].argsort()[::-1]] # sort iou
            # drop duplicate - detect
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]] # [0]:unique, [1]:indices
            # drop duplicate - label
#             matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        FP_cls[matches[:, 1].astype(int), i] = True
        if 'matches_tp' in locals():
            FP_cls[matches_tp[:, 1].astype(int), i] = False
            fp_cls_idx = np.setdiff1d(matches[:, 1], matches_tp[:, 1])
            matches_fp_cls = matches[np.isin(matches[:, 1], fp_cls_idx)]

    # FN, FP, IoU@.5
    matches_iou_x = torch.cat((torch.stack(condi_iou_x, 1), iou[condi_iou_x[0], condi_iou_x[1]][:, None]), 1).cpu().numpy()
    matches_iou_x = matches_iou_x[matches_iou_x[:, 2].argsort()[::-1]]
    matches_fn = matches_iou_x[np.unique(matches_iou_x[:, 0], return_index=True)[1]]
    matches_fp = matches_iou_x[np.unique(matches_iou_x[:, 1], return_index=True)[1]]

    if 'matches_tp' in locals():
        fn_idx = np.setdiff1d(matches_fn[:, 0], matches_tp[:, 0]) # fn label - tp label
        matches_fp = matches_fp[~np.isin(matches_fp[:, 1], matches_tp[:, 1])] # fp detect - tp detect 

    elif 'matches_tp' not in locals():
        fn_idx = matches_fn[:, 0]

    fp_cls_idx = np.where(FP_cls[:, 0]==True)[0]
    if fp_cls_idx.size > 0:
        matches_fp = matches_fp[~np.isin(matches_fp[:, 1], fp_cls_idx)]

    TP = torch.tensor(TP, dtype=torch.bool, device=iouv.device)
    FP_cls = torch.tensor(FP_cls, dtype=torch.bool, device=iouv.device)
    
    if 'matches_fp_cls' not in locals():
        matches_fp_cls = []
    if 'matches_tp' not in locals():
        matches_tp = []
    
    return matches_tp, matches_fp_cls, matches_fp, fn_idx


def merge_em_datas(matches_tp, matches_fp_cls, matches_fp, fn_idx, 
                   targets_cp, predn,
                   path, shape, names, em_datas):
    yolo_pts = load_txt_file(path)
    aimmo_pts = cvt_aimmo_pts(path, shape)
    for met_name, met in zip(['TP', 'FP_cls', 'FP', 'FN'], [matches_tp, matches_fp_cls, matches_fp, fn_idx]):
        if len(met):
            for met_i in met:
                tmp_dict = {}
                # FN -> dim 1, index
                if met_name == 'FN': 
                    # GT
                    tmp_dict['path'] = path.name
                    tmp_dict['label_i'] = met_i
#                     tmp_dict['label'] = aimmo_pts[int(met_i)][::2]
                    (tmp_dict['label_x1'], tmp_dict['label_y1']), (tmp_dict['label_x2'], tmp_dict['label_y2']) = aimmo_pts[int(met_i)][::2]
                    tmp_dict['yolo_pts'] = yolo_pts[int(met_i)][1:].tolist()
                    tmp_dict['label_cls'] = names[targets_cp[int(met_i)][0].item()]
                    tmp_dict['metrics'] = met_name
                # TP, FP_cls, FP -> dim 3, [label, detect, iou]
                else:
                    # GT
                    tmp_dict['path'] = path.name
                    tmp_dict['label_i'] = met_i[0]
#                     tmp_dict['label'] = aimmo_pts[int(met_i[0])][::2]
                    (tmp_dict['label_x1'], tmp_dict['label_y1']), (tmp_dict['label_x2'], tmp_dict['label_y2']) = aimmo_pts[int(met_i[0])][::2]
                    tmp_dict['yolo_pts'] = yolo_pts[int(met_i[0])][1:].tolist()
                    tmp_dict['label_cls'] = names[targets_cp[int(met_i[0])][0].item()]
                    tmp_dict['metrics'] = met_name
                    # Det
                    tmp_dict['det_i'] = met_i[1]
                    det = predn[int(met_i[1])][:4].int().tolist()
                    (tmp_dict['det_x1'], tmp_dict['det_y1']), (tmp_dict['det_x2'], tmp_dict['det_y2']) = [det[:2], det[2:]]
                    tmp_dict['det_cls'] = names[predn[int(met_i[1])][5].item()]
                    tmp_dict['conf'] = predn[int(met_i[1])][4].item()
                    tmp_dict['IoU'] = met_i[2]
#                     tmp_dict['cls_match'] = True if met_name == 'TP' else False
                    tmp_dict['cls_match'] = True if tmp_dict['label_cls'] == tmp_dict['det_cls'] else False
                em_datas.append(tmp_dict)
    return em_datas


def load_txt_file(yolo_img_path):
    txt_path = os.path.splitext(str(yolo_img_path).replace('images', 'labels'))[0]+'.txt'
    with open(txt_path, 'r') as f:
        txt_datas = f.readlines()
    txt_datas = [np.array(txt_data.strip().split(' '), dtype=float) for txt_data in txt_datas]    
    return txt_datas


def cvt_aimmo_pts(yolo_img_path, shape) -> list:
    txt_datas = load_txt_file(yolo_img_path)
#     txt_path = os.path.splitext(str(yolo_img_path).replace('images', 'labels'))[0]+'.txt'
#     with open(txt_path, 'r') as f:
#         txt_datas = f.readlines()
#     txt_datas = [np.array(txt_data.strip().split(' '), dtype=float) for txt_data in txt_datas]
    aimmo_pts = []
    for txt_data in txt_datas:
        aimmo_pts.append(make_yolo_point_to_aimmo_point(txt_data, H=shape[0], W=shape[1]))
    return aimmo_pts


def make_yolo_point_to_aimmo_point(points, H, W):
    '''
    description: Converting yolo coordinates to ammo coordinates
    input
        - points : yolo coordinates
            - type : list
            - e.g. : [cls, x, y, w, h]
        - H : Height of source image
        - W : Width of source image
    output: aimmo coordinates
        - type : list
    '''
    x, y, w, h = points[1:]
    x1 = W/2*(2*x-w)
    x2 = W/2*(2*x+w)
    x4, x3 = x1, x2
    y2 = H/2*(2*y-h)
    y3 = H/2*(2*y+h)
    y1, y4 = y2, y3
    aimmo_points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    aimmo_points = list(map(
        lambda points: [int(point) for point in points], aimmo_points))

    return aimmo_points
