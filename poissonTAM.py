import cv2
import pyzed.sl as sl
import numpy as np
import time
import rclpy
import rclpy.clock
from rclpy.node import Node
import rclpy.time
from std_msgs.msg import UInt8MultiArray
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import OccupancyGrid
import socket

import torch
from sam2.build_tam import build_tam_camera_predictor

checkpoint = "pytorch_env/lib/sam2/checkpoints/efficienttam_ti_512x512.pt"
model_cfg = "configs/sam2.1/efficienttam_ti_512x512.yaml"
predictor = build_tam_camera_predictor(model_cfg, checkpoint)

# Global variable
select_in_progress = False
origin_rect = (-1, -1)
points = []
labels = []
groups = []
ind = -1
mask = torch.empty((0,1), dtype=torch.float32)

msg1 = OccupancyGrid()
msg2 = Float32MultiArray()

IMAX = 120
JMAX = 120
DS = 0.0294

# Function that handles mouse events when interacting with the OpenCV window.
def on_mouse(event, x, y, flags, param):
    global select_in_progress, origin_rect, points, labels, cvImage, mask, group, groups, ind
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

class OccupancyGridPublisher(Node):
      
    def __init__(self):

        super().__init__('occupancy_grid_node')
        self.publisher_ = self.create_publisher(OccupancyGrid, 'occ_grid_topic', 1)
        
        msg1.header.stamp = self.get_clock().now().to_msg()
        msg1.header.frame_id = "world"
        msg1.info.width = IMAX
        msg1.info.height = JMAX
        msg1.info.resolution = DS
        msg1.info.origin.position.x = -1.75
        msg1.info.origin.position.y = -1.75
        msg1.info.origin.position.z = 0.0
        msg1.info.origin.orientation.w = 1.0
        msg1.info.origin.orientation.x = 0.0
        msg1.info.origin.orientation.y = 0.0
        msg1.info.origin.orientation.z = 0.0


class OpticalFlowPublisher(Node):
    
    def __init__(self):

        super().__init__('optical_flow_node')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'optical_flow_topic', 1)


def main(args=None):
    
    global points, labels, cvImage, mask, group, groups

    # Initialize ROS node
    rclpy.init(args=args)
    occupancy_grid_publisher = OccupancyGridPublisher()
    optical_flow_publisher = OpticalFlowPublisher()

    
    
    init = sl.InitParameters()
    #init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_resolution = sl.RESOLUTION.VGA
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.camera_fps = 100
    cam = sl.Camera()
    time.sleep(1)
    status = cam.open(init)
    while status != sl.ERROR_CODE.SUCCESS:
        status = cam.open(init)
    view = True

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    win_name = "Camera View"
    cv2.namedWindow(win_name, flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)
    print_camera_information(cam)
    print_help()

    #cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 30)

    key = ''

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

        err = cam.grab(runtime)
        cam.retrieve_image(mat, sl.VIEW.LEFT)  # Retrieve left image
        cvImRaw = mat.get_data()  # Convert sl.Mat to cv2.Mat
        cvImage = cv2.flip(cv2.flip(cvImRaw, 0), 1)
        #sqImage = cvImage[:,280:1000,:] # Crop 1280x720 Image to 720x720
        sqImage = cvImage[7:367,156:516,0:3] # Crop 672x376 Image to 360x360
        
        smImage = cv2.resize(sqImage, (IMAX,JMAX))
        prvs = cv2.cvtColor(smImage, cv2.COLOR_BGR2GRAY)
        flow0 = cv2.calcOpticalFlowFarneback(prvs, prvs, None, 0.5, 20, 15, 20, 7, 1.5, 0)
        arrOFj = np.reshape(flow0[...,0], IMAX*JMAX)
        arrOFi = np.reshape(flow0[...,1], IMAX*JMAX)
        arrOF = np.concatenate((arrOFi,arrOFj))
        arrOFV_old = 1.0 * arrOF
        arrOFV = 1.0 * arrOF
        t_flow = time.time()

        # Phase #1: #Initialization
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:  # Check that a new image is successfully acquired
            cam.retrieve_image(mat, sl.VIEW.LEFT)  # Retrieve left image
            cvImRaw = mat.get_data()  # Convert sl.Mat to cv2.Mat
            cvImage = cv2.flip(cv2.flip(cvImRaw, 0), 1)
            #sqImage = cvImage[:,280:1000,:] # Crop 1280x720 Image to 720x720
            sqImage = cvImage[7:367,156:516,0:3] # Crop 672x376 Image to 360x360
            
            predictor.load_first_frame(sqImage)
            while key != 13:  # for 'enter' key
                if key != -1:
                    group = key
                for g in range(mask.shape[0]):
                    mask_squeezed = mask[g,:].cpu().squeeze(0).squeeze(0)
                    sqImage[mask_squeezed>0,0] = 10 + 100*g % 255
                    sqImage[mask_squeezed>0,1] = 255 - 100*g % 255
                    sqImage[mask_squeezed>0,2] = 100
                    for g in range(len(points)):
                        for i in range(points[g].shape[0]):
                            sqImage[(points[g][i,1]-5):(points[g][i,1]+5),(points[g][i,0]-5):(points[g][i,0]+5),:] = 255
                            if labels[g][i] == 1:
                                sqImage[(points[g][i,1]-5):(points[g][i,1]+5),(points[g][i,0]-5):(points[g][i,0]+5),[0, 2]] = 0
                            else:
                                sqImage[(points[g][i,1]-5):(points[g][i,1]+5),(points[g][i,0]-5):(points[g][i,0]+5),[0, 1]] = 0
                cv2.imshow(win_name, sqImage)  # Display image
                key = cv2.waitKey(1)
        else:
            print("Error during capture : ", err)
        
        try:

            # Phase #2: Tracking
            while key != 113:  # for 'q' key
                if key != -1:
                    group = key
                if key == 118:
                    view = not view
                
                err = cam.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:  # Check that a new image is successfully acquired
                    
                    cam.retrieve_image(mat, sl.VIEW.LEFT)  # Retrieve left image
                    cvImRaw = mat.get_data()  # Convert sl.Mat to cv2.Mat
                    cvImage = cv2.flip(cv2.flip(cvImRaw, 0), 1)
                    #sqImage = cvImage[:,280:1000,:] # Crop 1280x720 Image to 720x720
                    sqImage = cvImage[7:367,156:516,0:3] # Crop 672x376 Image to 376x376
                    
                    group_inds, mask = predictor.track(sqImage)

                    sqGrid = cv2.cvtColor(sqImage, cv2.COLOR_BGR2GRAY)                
                    sqGrid *= 0
                    for g in range(len(group_inds)):
                        mask_squeezed = mask[g,:].cpu().squeeze(0).squeeze(0)
                        sqImage[mask_squeezed>0,0] = 10 + 100*g % 255
                        sqImage[mask_squeezed>0,1] = 255 - 100*g % 255
                        sqImage[mask_squeezed>0,2] = 100
                        sqGrid[mask_squeezed>0] = 127

                    smGrid = cv2.resize(sqGrid, (IMAX,JMAX), interpolation=cv2.INTER_LANCZOS4) # Downsample to IMAX x JMAX
                    
                    arrGrid = np.reshape(cv2.flip(smGrid, 0), IMAX*JMAX)
                    msg1.data = arrGrid
                    msg1.header.stamp = occupancy_grid_publisher.get_clock().now().to_msg()
                    occupancy_grid_publisher.publisher_.publish(msg1)

                    flow = cv2.calcOpticalFlowFarneback(prev=prvs, next=smGrid, flow=flow0, 
                                                        pyr_scale=0.5, levels=2, winsize=25,
                                                        iterations=10, poly_n=7, poly_sigma=1.5, flags=0)
                    
                    flow0 = flow
                    prvs = smGrid

                    dt_flow = time.time() - t_flow
                    t_flow = time.time()
                    arrOFj = np.reshape(cv2.flip(flow[...,0],0), IMAX*JMAX)
                    arrOFi = np.reshape(cv2.flip(flow[...,1],0), IMAX*JMAX) * -1.0
                    arrOF = np.concatenate((arrOFi,arrOFj))

                    # 2nd Order Low Pass Filter
                    wf = 20.0
                    kf = 1.0 - np.exp(-wf*dt_flow)
                    arrOFV_old *= 1.0 - kf
                    arrOFV_old += kf * arrOF / dt_flow
                    arrOFV *= 1.0 - kf
                    arrOFV += kf * arrOFV_old

                    msg2.data = arrOFV
                    optical_flow_publisher.publisher_.publish(msg2)
                    
                    bigImage = cv2.resize(sqImage, (720,720), interpolation=cv2.INTER_LANCZOS4)
                    cv2.imshow(win_name, bigImage)  # Display image
                
                else:
                    print("Error during capture : ", err)
                    break

                key = cv2.waitKey(1)
        
        except:
        
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
