#!/usr/bin/env python3
import time
import cv2
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray
import numpy as np
from scipy.ndimage import binary_dilation, convolve
# from .src.utils import *

class MapBuilderNode(Node):
    def __init__(self):

        super().__init__('grid_builder_node')
        self.subscription1 = self.create_subscription(OccupancyGrid, 'map_convert', self.mapping_callback, 1)
        self.subscription2 = self.create_subscription(PoseStamped, 'MacLane_pose_internal', self.state_update_callback, 1)
        self.binary_pub = self.create_publisher(OccupancyGrid, 'occupancy_grid', 1)
        #self.confidence_pub = self.create_publisher(OccupancyGrid, 'confidence_map', 1)
        self.optflow_pub = self.create_publisher(Float32MultiArray, 'optical_flow_topic', 1)

        self.HEIGHT = 60
        self.WIDTH = 60
        self.RESOLUTION = 0.0666666666666
        self.previous_confidence_grid = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.float32)

        self.rx = 0.0
        self.ry = 0.0
        self.rx_map = 0.0
        self.ry_map = 0.0
        self.occGrid0 = np.zeros((self.HEIGHT,self.WIDTH), dtype=np.uint8)
        self.flow0 = np.zeros((self.HEIGHT,self.WIDTH,2), dtype=np.float32)
        self.arrOFV_old = np.zeros(self.HEIGHT*self.WIDTH*2, dtype=np.float32)
        self.arrOFV = np.zeros(self.HEIGHT*self.WIDTH*2, dtype=np.float32)
        self.t = time.time()
        self.dt = 1.0e10

        self.tvl1 = cv2.optflow.createOptFlow_DualTVL1()
        self.tvl1.setUseInitialFlow(True)


    def state_update_callback(self, data):
            
        self.rx = data.pose.position.x
        self.ry = data.pose.position.y
    
    def mapping_callback(self, msg):
        
        self.dt = time.time() - self.t
        self.t = time.time()
        
        raw_map = np.array(msg.data, dtype=np.int8).reshape(self.HEIGHT, self.WIDTH)
        float_map = raw_map.astype(np.float32) / 127.0
        #buffered_map = binary_dilation(raw_map,iterations=1).astype(int)
        _, Confidence_values_conv = self.Filtered_Occupancy_Convolution_Masked(float_map, self.previous_confidence_grid)
        binary_map = self.thresholding_std(Confidence_values_conv, self.previous_confidence_grid)
        self.previous_confidence_grid = Confidence_values_conv
        #confidence = Confidence_values_conv/2

        binary_msg = self.numpy_to_occupancygrid_msg(binary_map)
        #confidence_msg = self.numpy_to_occupancygrid_msg(confidence)
        self.binary_pub.publish(binary_msg)
        #self.confidence_pub.publish(confidence_msg)

        self.old_bin_map = binary_map

        occGrid = (binary_map*255.0).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(prev=self.occGrid0, next=occGrid, flow=self.flow0, pyr_scale=0.5, levels=2, winsize=25, iterations=10, poly_n=7, poly_sigma=1.5, flags=0)
        #flow = self.tvl1.calc(self.occGrid0, occGrid, self.flow0)
        self.occGrid0 = occGrid
        self.flow0 = flow

        arrOFj = flow[...,0].flatten()
        arrOFi = flow[...,1].flatten()
        arrOF = np.concatenate((arrOFi,arrOFj)) / self.dt

        wf = 30.0
        kf = 1.0 - np.exp(-wf*self.dt)
        self.arrOFV_old *= 1.0 - kf
        self.arrOFV_old += kf * arrOF 
        self.arrOFV *= 1.0 - kf
        self.arrOFV += kf * self.arrOFV_old
  
        #print(np.amax(self.arrOFV) * self.RESOLUTION)

        optflow_msg = Float32MultiArray()
        optflow_msg.data = self.arrOFV
        self.optflow_pub.publish(optflow_msg)
        
        print(time.time()-self.t)

    def numpy_to_occupancygrid_msg(self, grid):
        
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.RESOLUTION
        msg.info.width = self.WIDTH
        msg.info.height = self.HEIGHT
        msg.info.origin = Pose()
        msg.info.origin.position.x = -(self.RESOLUTION*self.WIDTH) / 2.0
        msg.info.origin.position.y = -(self.RESOLUTION*self.HEIGHT) / 2.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        occGrid = (grid*127.0).astype(np.int8)
        msg.data = occGrid.flatten().tolist()
        
        return msg 
    
    def thresholding_std(self, conf_map, old_conf, T_hi=0.9, T_lo = 0.5):
        strong = conf_map >= T_hi #  Applying initial binary mask
        weak = (conf_map >= T_lo) & (conf_map < T_hi)
        grow_mask = (old_conf > T_lo) & weak # Promote weak pixels that had strong previouis signal
        binary_map = strong | grow_mask
        return binary_map.astype(np.uint8)
    
    def Filtered_Occupancy_Convolution_Masked(self, occupancy_data, C_old):
        
        # Shift Confidence Values Based on Egomotion
        drx = self.rx - self.rx_map
        dry = self.ry - self.ry_map
        self.rx_map = self.rx
        self.ry_map = self.ry
        di = int(np.round(dry / self.RESOLUTION, 0))
        dj = int(np.round(drx / self.RESOLUTION, 0))
        Confidence_values = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.float32)
        for i in range(self.HEIGHT):
            inew = i - di
            if inew > -1 and inew < self.HEIGHT:
                for j in range(self.WIDTH):
                    jnew = j - dj
                    if jnew > -1 and jnew < self.WIDTH:
                        Confidence_values[inew,jnew] = C_old[i,j]

        # Buffered binary convolution
        kernel = self.create_gaussian_decay_kernel(kernel_size=9, sigma=2.0, normalize=False)
        #kernel = self.create_square_decay_kernel(kernel_size=5, decay=0.15)
        buffered_binary_conv = convolve(occupancy_data, kernel, mode='constant', cval=0.0)

        C_plus = 1.0  # Confidence boost for strong responses
        C_minus = 0.0 # Confidence reduction for weak responses
        Th = 1.0  # Convolution Threshold for strong responses
        
        # Increase confidence in regions with  above_convolution_threshold
        beta1 = 4.0
        sig1 = 1.0 - np.exp(-beta1 * buffered_binary_conv * self.dt) # sigmoid
        mask1 = buffered_binary_conv > Th
        Confidence_values[mask1] *= 1.0 - sig1[mask1]
        Confidence_values[mask1] += sig1[mask1] * C_plus  # Boost strong responses

        # Decrease confidence in regions with below_convolution_threshold & inside visbility mask
        beta2 = 1.0
        sig2 = 1.0 - np.exp(-beta2 * self.dt)  # sigmoid
        mask2 = (buffered_binary_conv <= Th)# & grown_mask
        Confidence_values[mask2] *= 1.0 - sig2
        Confidence_values[mask2] += sig2 * C_minus
        
        return buffered_binary_conv, Confidence_values    
    

    def create_gaussian_decay_kernel(self, kernel_size: int, sigma: float = None, normalize: bool = False):
        """
        Creates a square kernel with values that decay with a 2D Gaussian.

        Args:
            kernel_size (int): Size of the kernel (must be odd).
            sigma (float, optional): Standard deviation of the Gaussian.
                                    If None, defaults to size / 6.
            normalize (bool, optional): Make kernel sum to 1

        Returns:
            np.ndarray: 2D Gaussian decay kernel (shape: size x size).
        """
        if kernel_size % 2 == 0: raise ValueError("Size must be an odd number.")
        if sigma is None: sigma = size / 6.0  # Common default to cover ~99% in the kernel

        # Create coordinate grid
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)) # Compute the 2D Gaussian
        if(normalize): kernel /= np.sum(kernel) # Normalize the kernel to sum to 1

        return kernel

    
    def create_square_decay_kernel(self, kernel_size=3, decay=0.2):
        """
        Create a square convolution kernel where each concentric Chebyshev ring decays linearly.

        Parameters:
            kernel_size (int): Size of the square kernel (must be odd, e.g., 3, 5, 7).
            decay (float): Linear decay applied per ring (e.g., 0.2 gives values 1.0, 0.8, 0.6...).

        Returns:
            kernel (np.ndarray): 2D convolution kernel.
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer.")
        
        center = kernel_size // 2
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

        for i in range(kernel_size):
            for j in range(kernel_size):
                ring = max(abs(i - center), abs(j - center))  # Chebyshev distance
                value = max(0.0, 1.0 - decay * ring)
                kernel[i, j] = value
        return kernel
    
if __name__=='__main__':
    rclpy.init(args=None)
    node=MapBuilderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
