import time
import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

imax = 200
jmax = 200

b = np.zeros((imax*jmax,1)) # Boundary Flag
h = np.zeros((imax*jmax,1)) + 36.0 # Safe Set
vx = np.zeros((imax*jmax,1)) # Guidance Field X
vy = np.zeros((imax*jmax,1)) # Guidance Field Y
f = np.zeros((imax*jmax,1)) # Forcing Function

# Border
h_border = 0.0
dh_border = 1.0
for i in range(0,imax):
    b[i*jmax+0] = True
    h[i*jmax+0] = h_border
    vx[i*jmax+0] = 0.0
    vy[i*jmax+0] = dh_border
    b[i*jmax+jmax-1] = True
    h[i*jmax+jmax-1] = h_border
    vx[i*jmax+jmax-1] = 0.0
    vy[i*jmax+jmax-1] = -dh_border
for j in range(0,jmax):
    b[0*jmax+j] = True
    h[0*jmax+j] = h_border
    vx[0*jmax+j] = dh_border
    vy[0*jmax+j] = 0.0
    b[(imax-1)*jmax+j] = True
    h[(imax-1)*jmax+j] = h_border
    vx[(imax-1)*jmax+j] = -dh_border
    vy[(imax-1)*jmax+j] = 0.0
vx[0*jmax+0] = dh_border*0.707
vy[0*jmax+0] = dh_border*0.707
vx[0*jmax+jmax-1] = dh_border*0.707
vy[0*jmax+jmax-1] = -dh_border*0.707
vx[(imax-1)*jmax+0] = -dh_border*0.707
vy[(imax-1)*jmax+0] = dh_border*0.707
vx[(imax-1)*jmax+jmax-1] = -dh_border*0.707
vy[(imax-1)*jmax+jmax-1] = -dh_border*0.707

# Boxes
h_box = 0.0
dh_box = 1.0
corners_box = np.zeros((3,4),'int')
corners_box[0,:] = [15,35,22,30]
corners_box[1,:] = [15,25,30,34]
corners_box[2,:] = [31,65,39,49]
for k in range(0,3):
    for i in range(corners_box[k,0],corners_box[k,1]+1):
        b[i*jmax+corners_box[k,2]] = True
        h[i*jmax+corners_box[k,2]] = h_box
        vx[i*jmax+corners_box[k,2]] = 0.0
        vy[i*jmax+corners_box[k,2]] = -dh_box
        b[i*jmax+corners_box[k,3]] = True
        h[i*jmax+corners_box[k,3]] = h_box
        vx[i*jmax+corners_box[k,3]] = 0.0
        vy[i*jmax+corners_box[k,3]] = dh_box
    for j in range(corners_box[k,2],corners_box[k,3]+1):
        b[corners_box[k,0]*jmax+j] = True
        h[corners_box[k,0]*jmax+j] = h_box
        vx[corners_box[k,0]*jmax+j] = -dh_box
        vy[corners_box[k,0]*jmax+j] = 0.0
        b[corners_box[k,1]*jmax+j] = True
        h[corners_box[k,1]*jmax+j] = h_box
        vx[corners_box[k,1]*jmax+j] = dh_box
        vy[corners_box[k,1]*jmax+j] = 0.0
    vx[corners_box[k,0]*jmax+corners_box[k,2]] = -dh_box*0.707
    vy[corners_box[k,0]*jmax+corners_box[k,2]] = -dh_box*0.707
    vx[corners_box[k,0]*jmax+corners_box[k,3]] = -dh_box*0.707
    vy[corners_box[k,0]*jmax+corners_box[k,3]] = dh_box*0.707
    vx[corners_box[k,1]*jmax+corners_box[k,2]] = dh_box*0.707
    vy[corners_box[k,1]*jmax+corners_box[k,2]] = -dh_box*0.707
    vx[corners_box[k,1]*jmax+corners_box[k,3]] = dh_box*0.707
    vy[corners_box[k,1]*jmax+corners_box[k,3]] = dh_box*0.707

# Delete that one overlapping segment
for i in range(16,25):
    b[i*jmax+30] = False

# Correct the interior corner
vx[25*jmax+30] = dh_box*0.707
vy[25*jmax+30] = dh_box*0.707

# Circle
h_circ = 0.0
dh_circ = 1.0
center = np.array([23,72])
for i in range(21,25):
    wgrad = ([i,69]-center) / np.sqrt(np.vdot([i,69]-center,[i,69]-center))
    b[i*jmax+69] = True
    h[i*jmax+69] = h_circ
    vx[i*jmax+69] = dh_circ * wgrad[0]
    vy[i*jmax+69] = dh_circ * wgrad[1]
    egrad = ([i,75]-center) / np.sqrt(np.vdot([i,75]-center,[i,75]-center))
    b[i*jmax+75] = True
    h[i*jmax+75] = h_circ
    vx[i*jmax+75] = dh_circ*egrad[0]
    vy[i*jmax+75] = dh_circ*egrad[1]
for j in range(70,74):
    sgrad = ([20,j]-center) / np.sqrt(np.vdot([20,i]-center,[20,i]-center))
    b[20*jmax+j] = True
    h[20*jmax+j] = h_circ
    vx[20*jmax+j] = dh_circ*sgrad[0]
    vy[20*jmax+j] = dh_circ*sgrad[1]
    ngrad = ([26,j]-center) / np.sqrt(np.vdot([26,j]-center,[26,j]-center))
    b[26*jmax+j] = True
    h[26*jmax+j] = h_circ
    vx[26*jmax+j] = dh_circ*ngrad[0]
    vy[26*jmax+j] = dh_circ*ngrad[1]

tic = time.time()

# Solve Interpolation Problem for vx
for j in range(1,jmax-1):
    for i1 in range(0,imax-1):
        if b[i1*jmax+j]:
            ileft = i1
            for i2 in range(ileft+1,imax):
                if b[i2*jmax+j]:
                    iright = i2
                    break
            di = iright - ileft
            if di > 1:
                for i3 in range(ileft,iright+1):
                    k = (i3-ileft) / di
                    vx[i3*jmax+j] = (1.0-k) * vx[ileft*jmax+j] + k * vx[iright*jmax+j]

# Solve Interpolation Problem for vy
for i in range(1,imax-1):
    for j1 in range(0,jmax-1):
        if b[i*jmax+j1]:
            jbottom = j1
            for j2 in range(jbottom+1,jmax):
                if b[i*jmax+j2]:
                    jtop = j2
                    break
            dj = jtop - jbottom
            if dj > 1:
                for j3 in range(jbottom,jtop+1):
                    k = (j3-jbottom) / dj
                    vy[i*jmax+j3] = (1.0-k) * vy[i*jmax+jbottom] + k * vy[i*jmax+jtop]

# Compute Forcing Function
for i in range(1,imax-1):
    for j in range(1,jmax-1):
        if not b[i*jmax+j]:
            f[i*jmax+j] = (vx[(i+1)*jmax+j] - vx[(i-1)*jmax+j] + vy[i*jmax+(j+1)] - vy[i*jmax+(j-1)]) / 2.0

# Solve Poisson's Equation
rms = 1e0
iter = 0
h_old = np.zeros((1,imax*jmax))

# Finite Difference Iteration for Specified Subgrid
def finiteDiff(bounds: tuple):
    i1 = bounds[0]
    i2 = bounds[1]
    for i in range(i1,i2):
        for j in range(0,jmax):
            if not b[i*jmax+j]:
                hnew[i*jmax+j] = (hnew[i*jmax+(j+1)] + hnew[i*jmax+(j-1)] + hnew[(i+1)*jmax+j] + hnew[(i-1)*jmax+j]) / 4.0 - f[i*jmax+j]


# Segment the Grid
segments = list()
threads = 100
for k in range(threads):
    segments.append([math.floor(k*imax/threads), math.ceil((k+1)*imax/threads)])

hnew = multiprocessing.Array('f', h)  
while rms > 1e-4:

    # Save Previous Solution
    h_old *= 0
    h_old += hnew
    
    # Run Finite Difference Method
    multiprocessing.Pool().map(finiteDiff, segments)

    # Compute Relative Change in Solution
    dh = hnew - h_old
    rms = np.linalg.norm(dh) / (imax*jmax)

    # Display Progress
    iter += 1
    if not np.mod(iter,1):
        print('Iteration: ',iter)
        print('Relative Error: ', rms)

toc = time.time()
dt = round(toc - tic,2)
print('Elapsed Time: ',dt,' seconds')

hfinal = np.zeros((imax,jmax))
for i in range(0,imax):
    for j in range(0,jmax):
        hfinal[i,j] = hnew[i*jmax+j]

fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
x = np.arange(0, imax, 1.0)
y = np.arange(0, jmax, 1.0)
x, y = np.meshgrid(x, y)
surf = ax1.plot_surface(x, y, hfinal, cmap='viridis')
fig1.colorbar(surf)

plt.show()
