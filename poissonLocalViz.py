import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import OccupancyGrid

import numpy as np
import matplotlib.pyplot as plt

imax = 120
jmax = 120
ds = 0.033333333
xc = -2.00
yc = -2.00
N = 10

x = np.zeros(N+1)
y = np.zeros(N+1)
yaw = np.zeros(N+1)
u = np.zeros(N+1)
v = np.zeros(N+1)

fig1, ax1 = plt.subplots(2,1,height_ratios=[30, 1])
plt.ion()
plt.tight_layout()

xvec = np.arange(0.0, jmax*ds, ds)
yvec = np.arange(0.0, imax*ds, ds)
X, Y = np.meshgrid(xvec, yvec)
H = np.zeros((imax,jmax), np.float32)

style = 'hot'
surf1 = ax1[0].pcolormesh(X+xc+x[0], Y+yc+y[0], H, vmin = -0.1, vmax = 0.7, cmap=style)
fig1.colorbar(surf1,cax=ax1[1], orientation='horizontal')
fig1.set_size_inches(8,10)

class MPCSubscriber(Node):
    
    def __init__(self):
        
        super().__init__('mpc_subscriber')
        self.subscription1 = self.create_subscription(Float32MultiArray, 'mpc_solution_topic', self.mpc_callback, 1)
        self.subscription2 = self.create_subscription(OccupancyGrid, 'occupancy_grid', self.occupancy_callback, 1)
        self.subscription3 = self.create_subscription(Float32MultiArray, 'safety_grid_topic', self.safety_callback, 1)
    
    def occupancy_callback(self, msg):

        global xc, yc, ds, imax, jmax

        xc = msg.info.origin.position.x + 0.38 * np.cos(yaw[0])
        yc = msg.info.origin.position.y + 0.38 * np.sin(yaw[0])
        imax = msg.info.width
        jmax = msg.info.height
        ds = msg.info.resolution

    def mpc_callback(self, msg1):

        global x, y, yaw, u, v, xc, yc

        for i in range(N+1):
            x[i] = msg1.data[3*i+0]
            y[i] = msg1.data[3*i+1]
            yaw[i] = msg1.data[3*i+2]
            u[i] = 1.0 * np.cos(yaw[i])
            v[i] = 1.0 * np.sin(yaw[i])
        
        xd = msg1.data[3*N + 3*(N+1) + 0]
        yd = msg1.data[3*N + 3*(N+1) + 1]

        # Update 3D Plot
        traj1 = ax1[0].quiver(x,y,u,v)
        traj2 = ax1[0].scatter(x[0],y[0], color='black', marker='s', s=100)
        traj3 = ax1[0].scatter(xd, yd, color='green', marker='*', s=250)
        ax1[0].set_xlim(xc+x[0],xc+x[0]+(imax-1)*ds)
        ax1[0].set_ylim(yc+y[0],yc+y[0]+(imax-1)*ds)
        ax1[0].set_aspect("equal")
        plt.waitforbuttonpress(0.001)

    def safety_callback(self, msg):

        global x, y, xc, yc, ds, imax, jmax
        
        xvec = np.arange(0.0, jmax*ds, ds)
        yvec = np.arange(0.0, imax*ds, ds)
        X, Y = np.meshgrid(xvec, yvec)
        H = np.reshape(msg.data, (imax,jmax))

        ax1[0].clear()
        surf1 = ax1[0].pcolormesh(X+xc+x[0], Y+yc+y[0], H, vmin = -0.2, vmax = 0.7, cmap=style)
        cont1 = ax1[0].contour(X+xc+x[0], Y+yc+y[0], H, levels=[0], linewidths={3}, cmap=style)


def main(args=None):

    rclpy.init(args=args)
    mpc_subscriber = MPCSubscriber()
    rclpy.spin(mpc_subscriber) 
    mpc_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()