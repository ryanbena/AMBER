import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import time

IMAX = 120
JMAX = 120
DS = 0.029412
N = 10

fig1, ax1 = plt.subplots(2,1,height_ratios=[30, 1])
plt.ion()
plt.tight_layout()

rx = np.zeros(N+1, np.float32)
ry = np.zeros(N+1, np.float32)

xvec = np.arange(0, JMAX*DS, DS)
yvec = np.arange(0, IMAX*DS, DS)
X, Y = np.meshgrid(xvec, yvec)

style = 'hot'
h = np.zeros((IMAX,JMAX), np.float32)
surf1 = ax1[0].pcolormesh(X, Y, h, vmin = -0.1, vmax = 0.7, cmap=style)
fig1.colorbar(surf1,cax=ax1[1], orientation='horizontal')

class CrazyfliePlotter(Node):

    def __init__(self):

        super().__init__('crazyflie_plotter')
        self.subscription1 = self.create_subscription(Float32MultiArray, 'mpc_solution_topic', self.mpc_callback, 1)
        self.subscription2 = self.create_subscription(Float32MultiArray, 'safety_grid_topic', self.safety_callback, 1)
        self.subscription1  # prevent unused variable warning
        self.subscription2  # prevent unused variable warning
    
    def mpc_callback(self, msg1):

        for i in range(N+1):
            rx[i] = msg1.data[6*i+0]
            ry[i] = msg1.data[6*i+1]

        rxd = msg1.data[6*(N+1) + 3*N + 0]
        ryd = msg1.data[6*(N+1) + 3*N + 1]

        # Update 3D Plot
        fig1.set_size_inches(8,10)
        traj1 = ax1[0].scatter(rx[0],ry[0], color='black', marker='s', s=100)
        traj2 = ax1[0].scatter(rxd, ryd, color='green', marker='*', s=250)
        traj3 = ax1[0].scatter(rx[1:], ry[1:], color='blue', marker='*', s=50)
        ax1[0].set_xlim(0,(JMAX-1)*DS)
        ax1[0].set_ylim(0,(IMAX-1)*DS)
        ax1[0].set_aspect("equal")
        plt.waitforbuttonpress(0.001)

    def safety_callback(self, msg2):

        h = np.reshape(msg2.data, (IMAX,JMAX))

        ax1[0].clear()
        surf1 = ax1[0].pcolormesh(X, Y, h, vmin = -0.1, vmax = 0.7, cmap=style)
        cont1 = ax1[0].contour(X, Y, h, levels=[0], linewidths={3}, cmap=style)
        ax1[0].set_xlim(0,(JMAX-1)*DS)
        ax1[0].set_ylim(0,(IMAX-1)*DS)        

def main(args=None):

    rclpy.init(args=args)
    cf_relay = CrazyfliePlotter()
    try:
        rclpy.spin(cf_relay)
    except:
        cf_relay.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    
    


