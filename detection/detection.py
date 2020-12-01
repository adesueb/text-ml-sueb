"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
from collections import OrderedDict

import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from detection import craft_utils
from detection import imgproc
from detection.craft import CRAFT


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def createBoundingBox(image, boxes):
    import cv2
    images = []
    for index, box in enumerate(boxes):
        cropped_image = image[int(box[0][1]):int(box[3][1]), int(box[0][0]):int(box[1][0])]
        cv2.imwrite('../cropped/crop{}.jpg'.format(index), cropped_image)
        images.append(cropped_image)
    return images


def test_net(net, image, text_threshold, link_threshold, low_text):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, False)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    return createBoundingBox(image, boxes)


def detect(filePath, modelPath):
    # load net
    net = CRAFT()  # initialize

    print('Loading weights from checkpoint (' + modelPath + ')')

    net.load_state_dict(copyStateDict(torch.load(modelPath)))
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

    net.eval()

    # load data
    image = imgproc.loadImage(filePath)

    images = test_net(net, image, 0.7, 0.4, 0.4)

    return images
