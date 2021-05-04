#!/usr/bin/env python

"""
Generate bottom-up attention features as a tsv file.
Can use multiple gpus, each produces a separate tsv file
that can be merged later (e.g. by using merge_tsv function).
Modify the load_image_ids script as necessary for your data location.
"""

# Example:
# ./tools/generate_tsv.py --gpu 0,1,2,3,4,5,6,7
# --cfg experiments/cfgs/faster_rcnn_end2end_resnet.ym
# --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt
# --out test2014_resnet101_faster_rcnn_genome.tsv
# --net data/faster_rcnn_m

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import glob

import caffe
import argparse
import pprint
import os
import sys
import base64
import numpy as np
import cv2
from multiprocessing import Process
import random

from PIL import Image


# Settings for the number of features per image.
# To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 100
MAX_BOXES = 100


def load_nlvr2_image_ids():
    split = []
    ext = ('.ras', '.xwd', '.bmp', '.jpe', '.jpg', '.jpeg', '.xpm',
           '.ief', '.pbm', '.tif', '.gif', '.ppm', '.xbm', '.tiff',
           '.rgb', '.pgm', '.png', '.pnm')
    image_file_name_list = [os.path.join(root, name)
                            for root, dirs, files in os.walk("/img")
                            for name in files
                            if name.endswith(tuple(ext))]
    image_ids = set()
    for filepath in image_file_name_list:
        filename, file_extension = os.path.splitext(
            filepath.split("/")[-1])
        image_id = filename
        if image_id not in image_ids:
            split.append((filepath, image_id))
            image_ids.add(image_id)
        else:
            assert image_id in image_ids, "image_id not unique"
    return split


def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):
    corrupted_im_return = {'image_id': image_id,
                           'image_h': 0,
                           'image_w': 0,
                           'num_boxes': 0, }
    im_file_name = im_file.split("/")[-1]
    try:
        img = Image.open(im_file).convert('RGB')
    except Exception as e:
        print(e)
        print("Corrupted image failed with Image.open: %s" % (im_file_name))
        return corrupted_im_return
    try:
        im = cv2.imread(im_file)
        if im is None:
            print("Corrupted image failed with cv2: %s" % (im_file))
            return corrupted_im_return
    except Exception as e:
        print(e)
        print("Corrupted image failed with cv2: %s" % (im_file))
        return corrupted_im_return

    try:
        print("Processing image_file: %s." % (im_file_name))
        scores, boxes, attr_scores, rel_scores = im_detect(net, im)
        # Keep the original boxes
        # don't worry about the regresssion bbox outputs
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        blobs, im_scales = _get_blobs(im, None)

        cls_boxes = rois[:, 1:5] / im_scales[0]

        cls_prob = net.blobs['cls_prob'].data
        pool5 = net.blobs['pool5_flat'].data
    except:
        print("Got exception when processing image_file: %s." % (im_file))
        return corrupted_im_return

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack(
            (cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep],
                                  cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]

    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': base64.b64encode(cls_boxes[keep_boxes]),
        'features': base64.b64encode(pool5[keep_boxes]),
        'confidence': base64.b64encode(max_conf[keep_boxes]),
        'soft_labels': base64.b64encode(scores[keep_boxes]),
    }


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from \
                                                  a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='./models/vg/ResNet-101/'
                                'faster_rcnn_end2end_final/test.prototxt',
                        type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--netDir', dest='caffemodelDir',
                        help='model saved folder',
                        default="./data/faster_rcnn_models/", type=str)
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default="./experiments/cfgs/"
                                "faster_rcnn_end2end_resnet.yml",
                        type=str)
    parser.add_argument('--prefix', help='dataset prefix',
                        default='nlvr2', type=str)
    parser.add_argument('--file_id', dest='file_id',
                        help='file id',
                        default=-1, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def save_to_npz(item, output_dir, prefix):
    image_id = item['image_id']
    outputfile = os.path.join(output_dir, prefix+image_id+".npz")

    image_w = float(item['image_w'])
    image_h = float(item['image_h'])

    item['num_boxes'] = int(item['num_boxes'])

    bboxes = np.frombuffer(base64.b64decode(item['boxes']),
                           dtype=np.float32
                           ).reshape((item['num_boxes'], -1))
    confidence = np.frombuffer(base64.b64decode(item['confidence']),
                               dtype=np.float64
                               ).reshape((item['num_boxes'], -1)
                                         ).astype(np.float32)
    soft_labels = np.frombuffer(base64.b64decode(item['soft_labels']),
                                dtype=np.float32
                                ).reshape((item['num_boxes'], -1))
    feature_string = item['features']
    try:
        decoded_feature = base64.b64decode(feature_string)
        curr_features = np.frombuffer(decoded_feature, dtype=np.float32)
        curr_features = curr_features.reshape((item['num_boxes'], -1))
    except:
        print("!!!!!!!!!!!!!!!!!!!!!Feature corrupted for imgid %s"
              % str(image_id))
        raise ValueError()
    box_width = bboxes[:, 2] - bboxes[:, 0]
    box_height = bboxes[:, 3] - bboxes[:, 1]
    scaled_width = box_width / image_w
    scaled_height = box_height / image_h
    scaled_x = bboxes[:, 0] / image_w
    scaled_y = bboxes[:, 1] / image_h

    box_width = box_width[..., np.newaxis]
    box_height = box_height[..., np.newaxis]
    scaled_width = scaled_width[..., np.newaxis]
    scaled_height = scaled_height[..., np.newaxis]
    scaled_x = scaled_x[..., np.newaxis]
    scaled_y = scaled_y[..., np.newaxis]

    normalized_bbox = np.concatenate((scaled_x, scaled_y,
                                      scaled_x + scaled_width,
                                      scaled_y + scaled_height,
                                      scaled_width, scaled_height), axis=1)
    np.savez_compressed(outputfile,
                        norm_bb=normalized_bbox.astype(np.float16),
                        features=curr_features.astype(np.float16),
                        conf=confidence.astype(np.float16),
                        soft_labels=soft_labels.astype(np.float16))


def generate_tsv(gpu_id, prototxt, weights, image_ids, out_dir, prefix=""):
    # First check if file exists, and if it is complete
    wanted_ids = set([image_id[1] for image_id in image_ids])
    found_ids = set()
    if os.path.exists(out_dir):
        files = glob.glob(os.path.join(out_dir, "{}*.npz".format(prefix)))
        for f in files:
            fname = os.path.basename(f)
            image_id = fname[6:-4]
            found_ids.add(image_id)
    missing = wanted_ids - found_ids
    if len(missing) == 0:
        print('GPU {:d}: already completed {:d}'
              .format(gpu_id, len(image_ids)))
    else:
        print('GPU {:d}: missing {:d}/{:d}'
              .format(gpu_id, len(missing), len(image_ids)))
    if len(missing) > 0:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        _t = {'misc': Timer()}
        count = 0
        for im_file, image_id in image_ids:
            if image_id in missing:
                try:
                    _t['misc'].tic()
                    det_features = get_detections_from_im(net, im_file,
                                                          image_id)
                    save_to_npz(det_features, out_dir, prefix)
                    _t['misc'].toc()
                    if (count % 100) == 0:
                        print('GPU {:d}: {:d}/{:d} {:.3f}s '
                              ' (projected finish: {:.2f} hours)'.format(
                                  gpu_id, count+1, len(missing),
                                  _t['misc'].average_time,
                                  _t['misc'].average_time
                                  * (len(missing)-count) / 3600))
                    count += 1
                except Exception:
                    # remove the corrupted file if created
                    outputfile = os.path.join(out_dir, prefix+image_id+".npz")
                    if os.path.exists(outputfile):
                        os.remove(outputfile)


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    image_ids = load_nlvr2_image_ids()
    random.seed(10)
    random.shuffle(image_ids)
    # Split image ids between gpus
    image_ids = [image_ids[i::len(gpus)] for i in range(len(gpus))]

    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    args.caffemodel = os.path.join(args.caffemodelDir,
                                   "resnet101_faster_rcnn_final.caffemodel")
    print(args.caffemodel)
    procs = []
    save_to_folder = "/output/"
    prefix = args.prefix + "_"
    for i, gpu_id in enumerate(gpus):
        p = Process(target=generate_tsv,
                    args=(gpu_id, args.prototxt, args.caffemodel,
                          image_ids[i], save_to_folder, prefix))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
