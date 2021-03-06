# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import dlib

from shapely.geometry import Point, Polygon

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS


# constants
WINDOW_NAME = "VOC detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--skip_frame", type=int, default=20, help='number of frames to skip')
    parser.add_argument("--min_score", type=int, default=0.5, help='minimum score to behave a box as TP')
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)

        skip_frame = args.skip_frame
        min_score = args.min_score

        ct = CentroidTracker(maxDisappeared=3)
        
        centroidX = 0
        centroidY = 0

        centroidX_arr = []
        centroidY_arr = []

        totalProducts = 0

        countProducts = 0
        countProducts_past = 0

        for countFrames, nums_pred, pred_boxes, pred_scores, vis_frame in tqdm.tqdm(demo.run_on_video(cam, skip_frame)):
            cv2.line(vis_frame, (width // 5, 0), (width // 5, 3 * height // 4), (0, 255, 255), 2)
            cv2.line(vis_frame, (4 * width // 5, 0), (4 * width // 5, 3 * height // 4), (0, 255, 255), 2)
            cv2.line(vis_frame, (width // 5, 3 * height // 4), (4 * width // 5, 3 * height // 4), (0, 255, 255), 2)

            cordinates = [(width // 5, 0), (4 * width // 5, 0), (4 * width // 5, 3 * height // 4), (width // 5, 3 * height // 4)]
            poly = Polygon(cordinates)

            rects = []
            tracks = []

            if countFrames % skip_frame == 0:
                print(f"+++++++++{countFrames}++++++++")
                print(nums_pred)
                print("+++++++++")
                centroidX_arr = []
                centroidY_arr = []
                for i in range(0, nums_pred):
                    confidence = pred_scores[i]
                    if (confidence >= min_score):
                        startX = pred_boxes[i][0]
                        startY = pred_boxes[i][1]
                        endX = pred_boxes[i][2]
                        endY = pred_boxes[i][3]
                        centroidX = int((endX + startX) / 2.0)
                        centroidY = int((endY + startY) / 2.0)
                        centroidX_arr.append(centroidX)
                        centroidY_arr.append(centroidY)
                        centroid = Point(centroidX, centroidY)
                        if poly.contains(centroid) == True:
                            box = [startX, startY, endX, endY]
                            rects.append(box)
                objects = ct.update(rects)
                # print("*************************")
                # print(objects)
                # print("*************************")

                
                
            countProducts = 0

            for (objectID, centroid) in objects.items():
                countProducts += 1
                text = f"ID {objectID}"
                cv2.putText(vis_frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(vis_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            if countProducts > countProducts_past:
                totalProducts += (countProducts - countProducts_past)
            
            # print(countProducts, countProducts_past)

            countProducts_past = countProducts
            cv2.putText(vis_frame, f'totalProducts: {str(totalProducts)}', (width // 2, height - 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (52, 202, 114), 2)
            cv2.putText(vis_frame, f'countProducts: {str(countProducts)}', (width // 2, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 167, 37), 2)
            # cv2.putText(vis_frame, f'frame: {str(countFrames)}', (width // 2, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 167, 37), 2)
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis_frame)
            countFrames += 1
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()


    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        fps = FPS().start()

#        if args.output:
#            if os.path.isdir(args.output):
#                output_fname = os.path.join(args.output, basename)
#                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
#            else:
#                output_fname = args.output
#            assert not os.path.isfile(output_fname), output_fname
        output_file = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('M','J','P','G'), frames_per_second, (width, height), isColor=True)
        assert os.path.isfile(args.video_input)

        skip_frame = args.skip_frame
        min_score = args.min_score

        ct = CentroidTracker(maxDisappeared=3)
        
        centroidX = 0
        centroidY = 0

        centroidX_arr = []
        centroidY_arr = []

        totalProducts = 0

        countProducts = 0
        countProducts_past = 0

        for countFrames, nums_pred, pred_boxes, pred_scores, vis_frame in tqdm.tqdm(demo.run_on_video(video, skip_frame), total=num_frames):
            cv2.line(vis_frame, (width // 5, 0), (width // 5, 3 * height // 4), (0, 255, 255), 2)
            cv2.line(vis_frame, (4 * width // 5, 0), (4 * width // 5, 3 * height // 4), (0, 255, 255), 2)
            cv2.line(vis_frame, (width // 5, 3 * height // 4), (4 * width // 5, 3 * height // 4), (0, 255, 255), 2)

            cordinates = [(width // 5, 0), (4 * width // 5, 0), (4 * width // 5, 3 * height // 4), (width // 5, 3 * height // 4)]
            poly = Polygon(cordinates)

            rects = []
            tracks = []

            if countFrames % skip_frame == 0:
                print(f"+++++++++{countFrames}++++++++")
                print(nums_pred)
                print("+++++++++")
                centroidX_arr = []
                centroidY_arr = []
                for i in range(0, nums_pred):
                    confidence = pred_scores[i]
                    if (confidence >= min_score):
                        startX = pred_boxes[i][0]
                        startY = pred_boxes[i][1]
                        endX = pred_boxes[i][2]
                        endY = pred_boxes[i][3]
                        centroidX = int((endX + startX) / 2.0)
                        centroidY = int((endY + startY) / 2.0)
                        centroidX_arr.append(centroidX)
                        centroidY_arr.append(centroidY)
                        centroid = Point(centroidX, centroidY)
                        if poly.contains(centroid) == True:
                            box = [startX, startY, endX, endY]
                            rects.append(box)
                objects = ct.update(rects)
                print("*************************")
                print(objects)
                print("*************************")

                
                
            countProducts = 0

            for (objectID, centroid) in objects.items():
                countProducts += 1
                text = f"ID {objectID}"
                cv2.putText(vis_frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(vis_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            if countProducts > countProducts_past:
                totalProducts += (countProducts - countProducts_past)
            
            print(countProducts, countProducts_past)

            countProducts_past = countProducts
            cv2.putText(vis_frame, f'totalProducts: {str(totalProducts)}', (width // 2, height - 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (52, 202, 114), 2)
            cv2.putText(vis_frame, f'countProducts: {str(countProducts)}', (width // 2, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 167, 37), 2)
            cv2.putText(vis_frame, f'frame: {str(countFrames)}', (width // 2, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 167, 37), 2)
            output_file.write(vis_frame)
            fps.update()
            # if args.output:
            #     print("numbers of predictions: ", nums_pred)
            #     print("bboxes predictions: ", pred_boxes)
            #     print("scores predictions: ", pred_scores)
            #     output_file.write(vis_frame)
            # else:
            #     cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
            #     cv2.imshow(basename, vis_frame)
            #     if cv2.waitKey(1) == 27:
            #         break  # esc to quit
        video.release()
        fps.stop()
        print(f"[INFO] elasped time: {fps.elapsed():.2f}")
        print(f"[INFO] approx. FPS: {fps.fps():.2f}")
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
