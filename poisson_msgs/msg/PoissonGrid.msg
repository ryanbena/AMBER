# PoissonGrid.msg


std_msgs/Header header          # Standard ROS2 Message Header

float32 resolution              # Uniform Grid Cell Size
uint32 width                    # Number of Columns
uint32 height                   # Number of Rows
geometry_msgs/Pose origin       # Bottom Left Coordinate

float32[] data                  # Poisson Grid Values