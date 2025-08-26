import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray

import cv2
import numpy as np
import time
import numpy as np

from load_lidar import *
from utils import *
from collections import deque

np.random.seed(42)

IMAX = 60
JMAX = 60
prvs = np.zeros((IMAX,JMAX), dtype=np.uint8)
flow0 = np.zeros((IMAX,JMAX,2), dtype=np.float32)
arrOFV_old = np.zeros(IMAX*JMAX*2, dtype=np.float32)
arrOFV = np.zeros(IMAX*JMAX*2, dtype=np.float32)
t_flow = time.time()

class LidarSaver(Node):
    
    def __init__(self):
        super().__init__('generate_masks_node')
        self.get_logger().info('STARTING')

        self.n_frames = 15
        self.frames_q = deque()
        self.foot_pts = np.empty(0)

        self.prev_clusters = []
        self.max_clusters = 80
        self.cid2idx = [-1] * self.max_clusters
        self.idx2cid = {}

        self.timer = Timer()
        self.timer.start_time("1 FRAME")

        # to discretize the world
        self.grid_length = IMAX
        # ignore outside of this
        self.radius = 2.0

        self.subscription = self.create_subscription(
            PointCloud2,
            "/utlidar/foot_position",
            self.callback_foot,
            1
        )
        time.sleep(0.5)
        self.subscription = self.create_subscription(
            PointCloud2,
            "/utlidar/cloud_deskewed",
            self.callback_lidar2,
            1
        )

        self.get_logger().info("created subscribers!")
        
        self.publisher = self.create_publisher(
            OccupancyGrid, 
            #'/occupancy_grid',
            '/map_convert', 
            1
        )

        self.publisher2 = self.create_publisher(
            Float32MultiArray, 
            'optical_flow_topic',
            1
        )

    def publish_occupancy_grid(self, mask):
        
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'
        
        grid_msg.info.resolution = (2.0 * self.radius) / self.grid_length
        grid_msg.info.width = mask.shape[1]
        grid_msg.info.height = mask.shape[0]

        grid_msg.info.origin = Pose()
        grid_msg.info.origin.position.x = -self.radius * 1.0
        grid_msg.info.origin.position.y = -self.radius * 1.0
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0 

        occGrid = mask.astype(np.int8) * 127
        grid_msg.data = occGrid.flatten().tolist()

        self.publisher.publish(grid_msg)
        
        global prvs, flow0, arrOFV_old, arrOFV, t_flow
        flow = cv2.calcOpticalFlowFarneback(prev=prvs, next=occGrid, flow=flow0, pyr_scale=0.5, levels=2, winsize=25, iterations=10, poly_n=7, poly_sigma=1.5, flags=0)
        flow0 = flow
        prvs = occGrid
        dt_flow = time.time() - t_flow
        t_flow = time.time()
        arrOFj = flow[...,0].flatten()
        arrOFi = flow[...,1].flatten()
        arrOF = np.concatenate((arrOFi,arrOFj)) / dt_flow
        wf = 100.0
        kf = 1.0 - np.exp(-wf*dt_flow)
        arrOFV_old *= 1.0 - kf
        arrOFV_old += kf * arrOF 
        arrOFV *= 1.0 - kf
        arrOFV += kf * arrOFV_old
        optflow_msg = Float32MultiArray()
        optflow_msg.data = arrOFV
        self.publisher2.publish(optflow_msg)

    def callback_lidar(self, msg):
        self.timer.end_time("1 FRAME")
        self.timer.start_time("1 FRAME")
        self.timer.start_time("PROCESS")
        
        points_it = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        self.frames_q.append([list(a) for a in list(points_it)])
        if len(self.frames_q) >= self.n_frames:
            pts_np = np.vstack(self.frames_q)
            self.process_and_save(pts_np)
            self.frames_q.popleft()
            self.timer.print()

        self.timer.end_time("PROCESS")
        
    def process_and_save(self, pts_np):
        dist2robo = self.radius
        res = self.grid_length
        pts_np = preprocess_data(pts_np.copy(), self.foot_pts.copy(), max_height=1.5, gd_buffer=0.03, maxdist=dist2robo)
        self.get_logger().info(f'num pts {pts_np.shape}')
        if pts_np.size:
            shadows, hulls = perform_2d_analysis_and_plot(pts_np, shadow_length=0)
            shadow_mask = generate_and_save_masks(shadows, hulls, dist2robo, res=res)
            self.publish_occupancy_grid(shadow_mask)

    
    def callback_lidar2(self, msg):
        self.timer.end_time("1 FRAME")
        self.timer.start_time("1 FRAME")
        self.timer.start_time("PROCESS")
    
        points_it = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        self.frames_q.append([list(a) for a in list(points_it)])
        pts_np = np.vstack(self.frames_q)
        self.frames_q.popleft()
        pts_np = preprocess_data(pts_np.copy(), self.foot_pts.copy(), max_height=1.0, gd_buffer=0.03, maxdist=self.radius)
        mask = np.zeros((self.grid_length, self.grid_length), dtype=bool)
        for pt in pts_np:
            i = int(round((pt[1] + self.radius) / (2.0*self.radius) * self.grid_length, 0))
            j = int(round((pt[0] + self.radius) / (2.0*self.radius) * self.grid_length, 0))
            if i > -1 and i < self.grid_length and j > -1 and j < self.grid_length: mask[i,j] = True
        self.publish_occupancy_grid(mask)
        

        self.timer.print()
        self.timer.end_time("PROCESS")
        
    def callback_foot(self, msg):
        points_it = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        self.foot_pts = np.array([list(a) for a in list(points_it)])

def main(args=None):
    rclpy.init(args=args)
    node = LidarSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
