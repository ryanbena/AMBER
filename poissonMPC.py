import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import cv2
import numpy as np
import matplotlib.pyplot as plt

imax = 120
jmax = 120
ds = 0.029412

fig1, ax1 = plt.subplots(1,2)
#fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
plt.ion()
plt.tight_layout()

x = np.array([])
y = np.array([])
yaw = np.array([])
u = np.array([])
v = np.array([])
h = np.zeros((imax*jmax), np.float32)
mpc = np.zeros(42, np.float32)

xvec = np.arange(0, imax*ds, ds)
yvec = np.arange(0, imax*ds, ds)
X, Y = np.meshgrid(xvec, yvec)

iter = 0

class MPCSubscriber(Node):
    
    def __init__(self):
        
        super().__init__('mpc_subscriber')
        self.subscription1 = self.create_subscription(Float32MultiArray, 'mpc_solution_topic', self.mpc_callback, 1)
        self.subscription2 = self.create_subscription(Float32MultiArray, 'safety_grid_topic', self.safety_callback, 1)
        self.subscription1  # prevent unused variable warning
        self.subscription2  # prevent unused variable warning

    def mpc_callback(self, msg1):

        global iter, h, mpc, mpc_new

        N = round((len(msg1.data)-2)/4) + 1
        x.resize(N)
        y.resize(N)
        yaw.resize(N)
        u.resize(N)
        v.resize(N)

        mpc_new = msg1.data

        for i in range(N):
            x[i] = msg1.data[2*i+0]
            y[i] = msg1.data[2*i+1]
            yaw[i] = 0.0 #msg1.data[3*i+2]
            u[i] = 1.0 * np.cos(yaw[i])
            v[i] = 1.0 * np.sin(yaw[i])

        # Update 3D Plot
        #ax1.clear()
        fig1.set_size_inches(14,8)
        #traj1 = ax1.quiver(x,y,u,v)
        traj1 = ax1[0].scatter(x[1:N-2],y[1:N-2], color='blue')
        traj2 = ax1[0].scatter(x[0],y[0], color='black', marker='s', s=100)
        traj3 = ax1[0].scatter(x[N-1],y[N-1], color='blue', marker='*', s=100)
        traj4 = ax1[0].scatter(1.75,2.75, color='green', marker='*', s=250)
        ax1[0].set_xlim(0,(imax-1)*ds)
        ax1[0].set_ylim(0,(imax-1)*ds)
        ax1[0].set_aspect("equal")
        plt.waitforbuttonpress(0.001)

    def safety_callback(self, msg2):

        global iter, h, mpc, mpc_new
        
        h_flat = msg2.data

        h = np.vstack((h, h_flat))
        mpc = np.vstack((mpc, mpc_new))

        h_new = np.reshape(h_flat, (imax,jmax))

        style = 'hot'

        ax1[0].clear()
        surf1 = ax1[0].pcolormesh(X, (imax-1)*ds-Y, h_new, vmin = -0.1, vmax = 0.7, cmap=style)
        cont1 = ax1[0].contour(X, (imax-1)*ds-Y, h_new, levels=[0], linewidths={3}, cmap=style)
        ax1[0].set_xlim(0,(imax-1)*ds)
        ax1[0].set_ylim(0,(imax-1)*ds)
        
        ax1[1].clear()
        surf2 = ax1[1].pcolormesh(X, (imax-1)*ds-Y, h_new, vmin = -0.1, vmax = 0.7, cmap=style)
        cont2 = ax1[1].contour(X, (imax-1)*ds-Y, h_new, levels=[0], linewidths={3}, cmap=style)
        #plt.colorbar(surf2)
        ax1[1].set_xlim(0,(imax-1)*ds)
        ax1[1].set_ylim(0,(imax-1)*ds)
        ax1[1].set_aspect("equal")
        
        iter += 1
       

def main(args=None):

    global h, mpc

    rclpy.init(args=args)

    mpc_subscriber = MPCSubscriber()
    try:
        rclpy.spin(mpc_subscriber) 
    except:
        np.save("h_test.npy", h)
        np.save("mpc_test.npy", mpc)
        
    mpc_subscriber.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()