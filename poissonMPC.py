import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import cv2
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
plt.ion()
plt.tight_layout()

x = np.array([])
y = np.array([])
first = True

class MPCSubscriber(Node):
    
    def __init__(self):
        
        super().__init__('mpc_subscriber')
        self.subscription = self.create_subscription(Float32MultiArray, 'mpc_solution_topic', self.mpc_callback, 1)
        self.subscription  # prevent unused variable warning

    def mpc_callback(self, msg):

        if(first):
            N = round((len(msg.data)-2)/4)+1
            x.resize(N)
            y.resize(N)
            first = False

        for i in range(N):
            x[i] = msg.data[2*i+0]
            y[i] = msg.data[2*i+1]
        
        # Update 3D Plot
        ax.clear()
        surf = ax.scatter(x,y)
        ax.set_xlim(0,3.5)
        ax.set_ylim(0,3.5)  
        plt.waitforbuttonpress(0.001)
        

def main(args=None):

    rclpy.init(args=args)

    mpc_subscriber = MPCSubscriber()
    rclpy.spin(mpc_subscriber) 
    mpc_subscriber.destroy_node()

    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()