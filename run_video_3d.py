import argparse
import logging
import time

import cv2
import numpy as np
#import matplotlib.animation as animation
#from io import BytesIO
#from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tf_pose import common

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.My_draw import plot_pose

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

if __name__ == '__main__':
    
    t = time.time()
    
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))

    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v') 
    writer = cv2.VideoWriter('./video/output3d.mp4', fourcc, fps, (640, 480))
    
    
    
    while cap.isOpened():
        ret_val, image = cap.read()
        if ret_val==True:
            logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            if not args.showBG:
                image = np.zeros(image.shape)
    
            logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            
            logger.info('3d lifting initialization.')
            poseLifting = Prob3dPose('./lifting/models/prob_model_params.mat')

            image_h, image_w = image.shape[:2]
            standard_w = 640
            standard_h = 480

            pose_2d_mpiis = []
            visibilities = []
            for human in humans:
                pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
                pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
                visibilities.append(visibility)

            pose_2d_mpiis = np.array(pose_2d_mpiis)
            visibilities = np.array(visibilities)
            transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
            pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)

            for i, single_3d in enumerate(pose_3d):
                fig = plot_pose(single_3d)
            
            fig.savefig("./tmp_im/im3d.jpg")
            img = cv2.imread("./tmp_im/im3d.jpg")

            cv2.imshow("result", img)
            writer.write(img)
            if cv2.waitKey(1) == 27:
                break
           
        else:
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    mytime = time.time() - t
    print("Total time: {}".format(mytime))
    

logger.debug('finished+')