# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
import time
import cv2
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
from tracker import *
import argparse

"""hyper parameters"""
use_cuda = True


def detect_cv2_camera(cfgfile, weightfile, source, show):

    model_name = (cfgfile.split('.')[-2]).split('/')[-1]
    m = Darknet(cfgfile)

    #m.print_network()
    m.load_weights(weightfile)
    #print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    # Create tracker object
    tracker = EuclideanDistTracker()
    cap = cv2.VideoCapture(source)
    times_infer, times_pipe = [], []

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    while True:
        ret, img = cap.read()

        if ret:
            img = cv2.resize(img, (1024,512))
            sized = cv2.resize(img, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)

            #result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

            t1 = time.time()
            t2 = time.time()

            times_infer.append(t1-t0)
            times_pipe.append(t2-t0)
                
            times_infer = times_infer[-20:]
            times_pipe = times_pipe[-20:]

            ms = sum(times_infer)/len(times_infer)*1000
            fps_infer = 1000 / (ms+0.00001)
            fps_pipe = 1000 / (sum(times_pipe)/len(times_pipe)*1000)

            boxes_ids = tracker.update(boxes[0])
            for box_id in boxes_ids:
                x, y, w, h, id = box_id
                cv2.putText(img, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Stream results
            if show:
                cv2.imshow(model_name, img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break             # 1 millisecond

            print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps_infer, fps_pipe))

        else:
            break        

    cap.release()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('--weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('--source', type=str, default='./output.mp4', help='path to video file.')
    parser.add_argument('--show', type=bool, default=True, help='display results')
    args = parser.parse_args()

    detect_cv2_camera(args.cfgfile, args.weightfile, args.source, args.show)
