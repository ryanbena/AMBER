import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray

import cv2
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

class SafetyGridSubscriber(Node):
    
    def __init__(self):
        
        super().__init__('safety_grid_subscriber')
        self.subscription = self.create_subscription(Float64MultiArray, 'safety_grid_topic', self.safety_callback, 10)
        self.subscription  # prevent unused variable warning
        cv2.namedWindow("Safety Grid")
        
        ds = 0.01
        x = np.arange(0, 1.2, ds)
        y = np.arange(0, 1.2, ds)
        X, Y = np.meshgrid(x, y)
        
        surf = ax.plot_surface(X, Y, X, cmap='viridis')
        fig.canvas.draw()
        plt.ion()
        plt.show()

    def safety_callback(self, msg):
        
        h = np.reshape(msg.data, (120,120))
        smImage = cv2.normalize(h, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        
        colorImage = cv2.cvtColor(smImage, cv2.COLOR_GRAY2BGR)
        grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
        for i in range(120):
            for j in range(120):
                colorImage[i,j,0] = 100 - grayImage[i,j] * (100/255)
                colorImage[i,j,1] = np.sqrt((255 - grayImage[i,j]) * grayImage[i,j])
                colorImage[i,j,2] = grayImage[i,j]
                
        bigImage = cv2.resize(colorImage, (720,720))
        cv2.imshow("Safety Grid", bigImage)  # Display image
        key = cv2.waitKey(1)
        
        ds = 0.01
        x = np.arange(0, 1.2, ds)
        y = np.arange(0, 1.2, ds)
        X, Y = np.meshgrid(x, y)
        ax.clear()
        surf = ax.plot_surface(X, Y, h, cmap='viridis')
        fig.canvas.draw()
        plt.waitforbuttonpress(0.1)
        

def main(args=None):

    

    rclpy.init(args=args)

    safety_grid_subscriber = SafetyGridSubscriber()
    rclpy.spin(safety_grid_subscriber) 
    safety_grid_subscriber.destroy_node()

    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()