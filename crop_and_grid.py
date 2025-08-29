import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

import numpy as np
import time

class LidarSaver(Node):
    
    def __init__(self):
        
        super().__init__('generate_masks_node')
        self.centroid = np.zeros(3)
        self.cos_yaw = 1.0
        self.sin_yaw = 0.0
        self.grid_length = 120 # to discretize the world
        self.minX = 0.38 # ignore inside of this
        self.minY = 0.18 # ignore inside of this
        self.maxXY = 2.0 # ignore outside of this
        self.minZ = 0.03 #ignore below this
        self.maxZ = 1.5 # ignore above this
        self.resolution = (2.0 * self.maxXY) / self.grid_length
        self.subscription = self.create_subscription(PoseStamped, "/utlidar/robot_pose", self.callback_pose, 1)
        time.sleep(1.0)
        self.subscription = self.create_subscription(PointCloud2, "/utlidar/cloud_deskewed", self.callback_lidar, 1)
        self.publisher = self.create_publisher(OccupancyGrid, '/map_convert', 1)

    def callback_pose(self, msg):
        
        self.centroid[0] = msg.pose.position.x
        self.centroid[1] = msg.pose.position.y
        self.centroid[2] = msg.pose.position.z
        self.sin_yaw = 2.0 * msg.pose.orientation.w * msg.pose.orientation.z 
        self.cos_yaw = 1.0 - 2.0 * msg.pose.orientation.z * msg.pose.orientation.z
    
    def callback_lidar(self, msg):

        tic = time.time()
        
        # Manipulate Data into Correct Format
        points_it = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points_ls = [list(a) for a in points_it]
        points_np = np.vstack(points_ls)

        # Crop Points
        xi = points_np[:, 0] - self.centroid[0]
        yi = points_np[:, 1] - self.centroid[1]
        zi = points_np[:, 2]
        xb =  self.cos_yaw * xi + self.sin_yaw * yi
        yb = -self.sin_yaw * xi + self.cos_yaw * yi
        in_range = (zi > self.minZ) & (zi < self.maxZ)
        in_range &= (np.abs(xi) < self.maxXY) & (np.abs(yi) < self.maxXY)
        in_range &= ((xb/self.minX)**2.0 + (yb/self.minY)**2.0) > 1.0
        points_np = points_np[in_range]

        # Create Mask
        mask = np.zeros((self.grid_length, self.grid_length), dtype=bool)
        i = np.round((points_np[:, 1] - self.centroid[1] + self.maxXY) / self.resolution).astype(int)
        j = np.round((points_np[:, 0] - self.centroid[0] + self.maxXY) / self.resolution).astype(int)
        try: mask[i, j] = True
        except: pass

        self.publish_occupancy_grid(mask)
        
        toc = time.time()
        print(toc - tic)

    def publish_occupancy_grid(self, mask):
        
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = mask.shape[1]
        grid_msg.info.height = mask.shape[0]
        grid_msg.info.origin = Pose()
        grid_msg.info.origin.position.x = -self.maxXY
        grid_msg.info.origin.position.y = -self.maxXY
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0 
        occGrid = mask.astype(np.int8) * 127
        grid_msg.data = occGrid.flatten().tolist()
        self.publisher.publish(grid_msg)


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
