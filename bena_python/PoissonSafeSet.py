import time
import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

imax = 100
jmax = 100

b = np.ones((imax,jmax)) # Safety Flag
sdf = np.zeros((imax,jmax)) # Signed Distance Function
h = np.zeros((imax,jmax)) + 18.0 # Safe Set
vx = np.zeros((imax,jmax)) # Guidance Field X
vy = np.zeros((imax,jmax)) # Guidance Field Y
f = np.zeros((imax,jmax)) # Forcing Function

# Border
h_border = 0.0
dh_border = 1.0
for i in range(0,imax):
    b[i,0] = 0
    h[i,0] = h_border
    vx[i,0] = 0.0
    vy[i,0] = dh_border
    b[i,jmax-1] = 0
    h[i,jmax-1] = h_border
    vx[i,jmax-1] = 0.0
    vy[i,jmax-1] = -dh_border
for j in range(0,jmax):
    b[0,j] = 0
    h[0,j] = h_border
    vx[0,j] = dh_border
    vy[0,j] = 0.0
    b[imax-1,j] = 0
    h[imax-1,j] = h_border
    vx[imax-1,j] = -dh_border
    vy[imax-1,j] = 0.0
vx[0,0] = dh_border*0.707
vy[0,0] = dh_border*0.707
vx[0,jmax-1] = dh_border*0.707
vy[0,jmax-1] = -dh_border*0.707
vx[imax-1,0] = -dh_border*0.707
vy[imax-1,0] = dh_border*0.707
vx[imax-1,jmax-1] = -dh_border*0.707
vy[imax-1,jmax-1] = -dh_border*0.707

# Boxes
h_box = 0.0
dh_box = 1.0
corners_box = np.zeros((3,4),'int')
corners_box[0,:] = [15,35,22,30]
corners_box[1,:] = [15,25,30,34]
corners_box[2,:] = [31,65,39,49]
for k in range(0,3):
    b[corners_box[k,0]:corners_box[k,1],corners_box[k,2]:corners_box[k,3]] = -1
    for i in range(corners_box[k,0],corners_box[k,1]+1):
        b[i,corners_box[k,2]] = 0
        h[i,corners_box[k,2]] = h_box
        vx[i,corners_box[k,2]] = 0.0
        vy[i,corners_box[k,2]] = -dh_box
        b[i,corners_box[k,3]] = 0
        h[i,corners_box[k,3]] = h_box
        vx[i,corners_box[k,3]] = 0.0
        vy[i,corners_box[k,3]] = dh_box
    for j in range(corners_box[k,2],corners_box[k,3]+1):
        b[corners_box[k,0],j] = 0
        h[corners_box[k,0],j] = h_box
        vx[corners_box[k,0],j] = -dh_box
        vy[corners_box[k,0],j] = 0.0
        b[corners_box[k,1],j] = 0
        h[corners_box[k,1],j] = h_box
        vx[corners_box[k,1],j] = dh_box
        vy[corners_box[k,1],j] = 0.0
    vx[corners_box[k,0],corners_box[k,2]] = -dh_box*0.707
    vy[corners_box[k,0],corners_box[k,2]] = -dh_box*0.707
    vx[corners_box[k,0],corners_box[k,3]] = -dh_box*0.707
    vy[corners_box[k,0],corners_box[k,3]] = dh_box*0.707
    vx[corners_box[k,1],corners_box[k,2]] = dh_box*0.707
    vy[corners_box[k,1],corners_box[k,2]] = -dh_box*0.707
    vx[corners_box[k,1],corners_box[k,3]] = dh_box*0.707
    vy[corners_box[k,1],corners_box[k,3]] = dh_box*0.707

# Delete that one overlapping segment
b[16:25,30] = -1
# Correct the interior corner
vx[25,30] = dh_box*0.707
vy[25,30] = dh_box*0.707

# Circle
h_circ = 0.0
dh_circ = 1.0
center = np.array([23,72])
b[21:26,70:75] = -1
for i in range(21,26):
    wgrad = ([i,69]-center) / np.sqrt(np.vdot([i,69]-center,[i,69]-center))
    b[i,69] = 0
    h[i,69] = h_circ
    vx[i,69] = dh_circ * wgrad[0]
    vy[i,69] = dh_circ * wgrad[1]
    egrad = ([i,75]-center) / np.sqrt(np.vdot([i,75]-center,[i,75]-center))
    b[i,75] = 0
    h[i,75] = h_circ
    vx[i,75] = dh_circ*egrad[0]
    vy[i,75] = dh_circ*egrad[1]
for j in range(70,75):
    sgrad = ([20,j]-center) / np.sqrt(np.vdot([20,i]-center,[20,i]-center))
    b[20,j] = 0
    h[20,j] = h_circ
    vx[20,j] = dh_circ*sgrad[0]
    vy[20,j] = dh_circ*sgrad[1]
    ngrad = ([26,j]-center) / np.sqrt(np.vdot([26,j]-center,[26,j]-center))
    b[26,j] = 0
    h[26,j] = h_circ
    vx[26,j] = dh_circ*ngrad[0]
    vy[26,j] = dh_circ*ngrad[1]

tic = time.time()

# Solve SDF
for i in range(0,imax):
    for j in range(0,jmax):
        if b[i,j]:
            min_dist = math.sqrt(imax**2+jmax**2)
            min_i = 0
            min_j = 0
            for x in range(1,math.ceil(imax/2)):
                if (x > min_dist):
                    break
                for y in range(1,math.ceil(jmax/2)):
                    if (y > min_dist):
                        break
                    dist = math.sqrt(x**2+y**2)
                    if not b[i+x,j+y] and (dist<min_dist):
                        min_dist = dist
                        min_i = i
                        min_j = j
                    if not b[i+x,j-y] and (dist<min_dist):
                        min_dist = dist
                        min_i = i
                        min_j = j
                    if not b[i-x,j+y] and (dist<min_dist):
                        min_dist = dist
                        min_i = i
                        min_j = j
                    if not b[i-x,j-y] and (dist<min_dist):
                        min_dist = dist
                        min_i = i
                        min_j = j
            sdf[i,j] = min_dist * b[min_i,min_j]

toc = time.time()
dt1 = round(toc - tic,2)

tic = time.time()

# Solve Interpolation Problem for vx
for j in range(1,jmax-1):
    for i1 in range(0,imax-1):
        if not b[i1,j]:
            ileft = i1
            for i2 in range(ileft+1,imax):
                if not b[i2,j]:
                    iright = i2
                    break
            di = iright - ileft
            if di > 1:
                for i3 in range(ileft,iright+1):
                    k = (i3-ileft) / di
                    vx[i3,j] = (1.0-k) * vx[ileft,j] + k * vx[iright,j]

# Solve Interpolation Problem for vy
for i in range(1,imax-1):
    for j1 in range(0,jmax-1):
        if not b[i,j1]:
            jbottom = j1
            for j2 in range(jbottom+1,jmax):
                if not b[i,j2]:
                    jtop = j2
                    break
            dj = jtop - jbottom
            if dj > 1:
                for j3 in range(jbottom,jtop+1):
                    k = (j3-jbottom) / dj
                    vy[i,j3] = (1.0-k) * vy[i,jbottom] + k * vy[i,jtop]

# Compute Forcing Function
for i in range(1,imax-1):
    for j in range(1,jmax-1):
        if b[i,j]:
            f[i,j] = (vx[i+1,j] - vx[i-1,j] + vy[i,j+1] - vy[i,j-1]) / 8.0

# Solve Poisson's Equation
rms = 1e0
iter = 0
h *= 0
h += sdf
h_old = np.zeros((imax,jmax))
while rms > 5e-5:
    
    iter += 1
    h_old *= 0
    h_old += h
    
    for i in range(1,imax-1):
        for j in range(1,jmax-1):
            if b[i,j]:
                h[i,j] = (h[i,j+1] + h[i,j-1] + h[i+1,j] + h[i-1,j]) / 4.0 - f[i,j]

    for i in range(imax-1,1,-1):
        for j in range(jmax-1,1,-1):
            if b[i,j]:
                h[i,j] = (h[i,j+1] + h[i,j-1] + h[i+1,j] + h[i-1,j]) / 4.0 - f[i,j]

    dh = h - h_old
    rms = np.linalg.norm(dh,'fro') / (imax*jmax)
    if not np.mod(iter,1):
        print('Iteration: ',iter)
        print('Relative Error: ', rms)

toc = time.time()
dt2 = round(toc - tic,2)

np.savetxt('safetyFlag.csv', b, delimiter=',')
np.savetxt('forcingFun.csv', f, delimiter=',')

print('SDF Elapsed Time: ',dt1,' seconds')
print('Poissson Elapsed Time: ',dt2,' seconds')

x = np.arange(0, 100, 1.0)
y = np.arange(0, 100, 1.0)
x, y = np.meshgrid(x, y)

fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax1.plot_surface(x, y, h, cmap='viridis')
colorbar1 = fig1.colorbar(surf1)

fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
surf2 = ax2.plot_surface(x, y, sdf, cmap='viridis')
colorbar2 = fig2.colorbar(surf2)

plt.show()
