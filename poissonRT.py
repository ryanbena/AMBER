import cv2
import pyzed.sl as sl
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import UInt8MultiArray

import torch

from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_camera_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Global variable
camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1
led_on = False
select_in_progress = False
origin_rect = (-1, -1)
points = []
labels = []
groups = []
ind = -1
mask = torch.empty((0,1), dtype=torch.float32)

# Function that handles mouse events when interacting with the OpenCV window.
def on_mouse(event, x, y, flags, param):
    global select_in_progress, selection_rect, origin_rect, points, labels, cvImage, mask, group, groups, ind
    if event == cv2.EVENT_LBUTTONDOWN:
        origin_rect = (x, y)
        select_in_progress = True
    elif event == cv2.EVENT_LBUTTONUP:
        select_in_progress = False
        new_point = np.array([[x, y]], dtype=np.int32)
        new_label = np.array([1], dtype=np.int32) 
        if group not in groups:
            groups.append(group)
            points.append(np.empty((0, 2), dtype=np.int32))
            labels.append(np.empty(0, dtype=np.int32))
            ind+=1
        points[ind] = np.append(points[ind], new_point, axis=0)
        labels[ind] = np.append(labels[ind], new_label)
        _,_,mask = predictor.add_new_prompt(frame_idx=0, obj_id=groups[ind], points=new_point,labels=new_label)
    elif event == cv2.EVENT_RBUTTONDOWN:
        origin_rect = (x, y)
        select_in_progress = True
    elif event == cv2.EVENT_RBUTTONUP:
        select_in_progress = False
        new_point = np.array([[x, y]], dtype=np.int32)
        new_label = np.array([0], dtype=np.int32) 
        if group not in groups:
            groups.append(group)
            points.append(np.empty((0, 2), dtype=np.int32))
            labels.append(np.empty(0, dtype=np.int32))
            ind+=1
        points[ind] = np.append(points[ind], new_point, axis=0)
        labels[ind] = np.append(labels[ind], new_label)
        _,_,mask = predictor.add_new_prompt(frame_idx=0, obj_id=groups[ind], points=new_point,labels=new_label)


checkpoint = "pytorch_env/lib/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_camera_predictor(model_cfg, checkpoint)


class OccupancyGridPublisher(Node):
    def __init__(self):
        super().__init__('occupancy_grid_node')
        self.publisher_ = self.create_publisher(UInt8MultiArray, 'occ_grid_topic', 10)

def main(args=None):
    
    global points, labels, cvImage, mask, group, groups

    # Initialize ROS node
    rclpy.init(args=args)
    occupancy_grid_publisher = OccupancyGridPublisher()

    init = sl.InitParameters()
    init.camera_fps = 60
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.NONE
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit()
    view = True

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    win_name = "Camera View"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)
    print_camera_information(cam)
    print_help()

    cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 30)

    key = ''

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        
        # Phase #1: #Initialization
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:  # Check that a new image is successfully acquired
            cam.retrieve_image(mat, sl.VIEW.LEFT)  # Retrieve left image
            cvImage = mat.get_data()  # Convert sl.Mat to cv2.Mat
            predictor.load_first_frame(cvImage[:,:,0:3])
            while key != 13:  # for 'enter' key
                if key != -1:
                    group = key
                for g in range(mask.shape[0]):
                    mask_squeezed = mask[g,:].cpu().squeeze(0).squeeze(0)
                    cvImage[mask_squeezed>0,0] = 10 + 100*g % 255
                    cvImage[mask_squeezed>0,1] = 255 - 100*g % 255
                    cvImage[mask_squeezed>0,2] = 100
                    for g in range(len(points)):
                        for i in range(points[g].shape[0]):
                            cvImage[(points[g][i,1]-5):(points[g][i,1]+5),(points[g][i,0]-5):(points[g][i,0]+5),:] = 255
                            if labels[g][i] == 1:
                                cvImage[(points[g][i,1]-5):(points[g][i,1]+5),(points[g][i,0]-5):(points[g][i,0]+5),[0, 2]] = 0
                            else:
                                cvImage[(points[g][i,1]-5):(points[g][i,1]+5),(points[g][i,0]-5):(points[g][i,0]+5),[0, 1]] = 0
                cv2.imshow(win_name, cvImage)  # Display image
                key = cv2.waitKey(1)
        else:
            print("Error during capture : ", err)

        # Phase #2: Tracking
        while key != 113:  # for 'q' key
            if key != -1:
                group = key
            if key == 118:
                view = not view
            
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:  # Check that a new image is successfully acquired
                
                cam.retrieve_image(mat, sl.VIEW.LEFT)  # Retrieve left image
                cvImage = mat.get_data()  # Convert sl.Mat to cv2.Mat
                group_inds, mask = predictor.track(cvImage[:,:,0:3])
            
                cvGrid = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
                cvGrid *= 0

                for g in range(len(group_inds)):
                    mask_squeezed = mask[g,:].cpu().squeeze(0).squeeze(0)
                    cvImage[mask_squeezed>0,0] = 10 + 100*g % 255
                    cvImage[mask_squeezed>0,1] = 255 - 100*g % 255
                    cvImage[mask_squeezed>0,2] = 100
                    cvGrid[mask_squeezed>0] = 255
                
                sqGrid = cvGrid[:,280:1000] # Crop 1280x720 to 720x720 Square
                smGrid = cv2.resize(sqGrid, (120,120)) # Downsample to 120x120
                arrGrid = np.reshape(smGrid, 120*120)

                msg = UInt8MultiArray()
                msg.data = arrGrid
                occupancy_grid_publisher.publisher_.publish(msg)

                if view:
                    cv2.imshow(win_name, cvImage)  # Display image
                else:
                    cv2.imshow(win_name, cv2.flip(cv2.flip(cvImage, 0), 1))  # Display image
            
            else:
                print("Error during capture : ", err)
                break

            key = cv2.waitKey(1)
        
        occupancy_grid_publisher.destroy_node()
        rclpy.shutdown()

        cv2.destroyAllWindows()
        cam.close()



# Display camera information
def print_camera_information(cam):
    cam_info = cam.get_camera_information()
    print("ZED Model                 : {0}".format(cam_info.camera_model))
    print("ZED Serial Number         : {0}".format(cam_info.serial_number))
    print("ZED Camera Firmware       : {0}/{1}".format(cam_info.camera_configuration.firmware_version,
                                                       cam_info.sensors_configuration.firmware_version))
    print("ZED Camera Resolution     : {0}x{1}".format(round(cam_info.camera_configuration.resolution.width, 2),
                                                       cam.get_camera_information().camera_configuration.resolution.height))
    print("ZED Camera FPS            : {0}".format(int(cam_info.camera_configuration.fps)))


# Print help
def print_help():
    print("\n\nCamera controls hotkeys:")
    print("* Increase camera settings value:  '+'")
    print("* Decrease camera settings value:  '-'")
    print("* Toggle camera settings:          's'")
    print("* Toggle camera LED:               'l' (lower L)")
    print("* Reset all parameters:            'r'")
    print("* Reset exposure ROI to full image 'f'")
    print("* Use mouse to select an image area to apply exposure (press 'a')")
    print("* Exit :                           'q'\n")

if __name__ == "__main__":
    main()
