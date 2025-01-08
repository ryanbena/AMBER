import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

# Read the CSV files
with open('sdf_safe_set.csv', 'r') as file:
    reader1 = csv.reader(file)
    data1 = list(reader1)

with open('poisson_safe_set.csv', 'r') as file:
    reader2 = csv.reader(file)
    data2 = list(reader2)

with open('guidance_field_x.csv', 'r') as file:
    reader3 = csv.reader(file)
    data3 = list(reader3)

with open('guidance_field_y.csv', 'r') as file:
    reader4 = csv.reader(file)
    data4 = list(reader4)

with open('occupancy_map.csv', 'r') as file:
    reader5 = csv.reader(file)
    data5 = list(reader5)

with open('forcing_function.csv', 'r') as file:
    reader6 = csv.reader(file)
    data6 = list(reader6)

# Convert the data to a NumPy array
sdf = np.array(data1, dtype='double')
h = np.array(data2, dtype='double')
vx = np.array(data3, dtype='double')
vy = np.array(data4, dtype='double')
b = np.array(data5, dtype='double')
f = np.array(data6, dtype='double')

# Perform Computations on Data
grady, gradx = np.gradient(h, edge_order=1)

ds = 0.01
x = np.arange(0, 4, ds)
y = np.arange(0, 4, ds)
X, Y = np.meshgrid(x, y)

fig1, ax1 = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"})
surf1 = ax1[0].plot_surface(X, Y, sdf, cmap='viridis')
surf2 = ax1[1].plot_surface(X, Y, h, cmap='viridis')
#surf1 = ax1[0].contour(X, Y, sdf, [-1.0,0.0,0.25,1.0], cmap='viridis')
#surf2 = ax1[1].contour(X, Y, h, cmap='viridis')

fig2, ax2 = plt.subplots(nrows=1, ncols=2)
quiver1 = ax2[0].quiver(X, Y, vy, vx)
ax2[0].title.set_text("Guidance Field")
quiver2 = ax2[1].quiver(X, Y, gradx, grady)
level1 = ax2[1].contour(X, Y, b, [-10.0,0.0,10.0], cmap='viridis')
level2 = ax2[1].contour(X, Y, h, [-10.0,0.0,10.0], cmap='autumn')
rect1 = plt.Rectangle((0.88-ds, 0.60-ds), 0.32+ds, 0.80+ds, facecolor="black", alpha=0.5)
rect2 = plt.Rectangle((1.20-ds, 0.60-ds), 0.16+ds, 0.40+ds, facecolor="black", alpha=0.5)
rect3 = plt.Rectangle((1.56-ds, 1.24-ds), 0.40+ds, 1.36+ds, facecolor="black", alpha=0.5)
rect4 = plt.Rectangle((2.76-ds, 0.84-ds), 0.28+ds, 0.28+ds, facecolor="black", alpha=0.5)
ax2[1].add_patch(rect1)
ax2[1].add_patch(rect2)
ax2[1].add_patch(rect3)
ax2[1].add_patch(rect4)

ax2[1].title.set_text("Gradient Field")

fig3, ax3 = plt.subplots()
quiver3 = ax3.quiver(X, Y, b, b)

fig4, ax4 = plt.subplots(subplot_kw={"projection": "3d"})
surf3 = ax4.plot_surface(X, Y, f)

#surf2 = ax1[1].plot_surface(x, y, h, cmap='viridis')
#surf3 = ax1[1, 0].quiver(x, y, vx, vy)
#surf4 = ax1[1, 1].plot_surface(x, y, h, cmap='viridis')

#fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
#contour1 = ax2.contour(x, y, h, np.linspace(-1.0, 2.0, 31))
#colorbar2 = fig2.colorbar(contour1)

plt.show()