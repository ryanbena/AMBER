import numpy as np
import matplotlib.pyplot as plt

h_data = np.load('h_test.npy')
mpc_data = np.load('mpc_test.npy')

imax = 120
jmax = 120
ds = 0.029412
dt = 0.085
N = 10
lenMPC = 3*N + 3*(N+1) + 2

fig1, ax1 = plt.subplots(2,3,height_ratios=[30, 1])
fig1.set_size_inches(16,10)
plt.tight_layout()

xvec = np.arange(0, imax*ds, ds)
yvec = np.arange(0, imax*ds, ds)
X, Y = np.meshgrid(xvec, yvec)

style = 'hot'

h_new = np.zeros((imax,jmax), np.float32)
mpc_new = np.zeros(lenMPC, np.float32)
surf1 = ax1[0,0].pcolormesh(X, (imax-1)*ds-Y, h_new, vmin = -0.1, vmax = 0.7, cmap=style)
fig1.colorbar(surf1,cax=ax1[1,0], orientation='horizontal')

x = np.zeros(N+1, np.float32)
y = np.zeros(N+1, np.float32)
yaw = np.zeros(N+1, np.float32)
u = np.zeros(N+1, np.float32)
v = np.zeros(N+1, np.float32)

i_start = 3600
i_stop = 4800
#print(h_data.shape[0])
#i_start = int(h_data.shape[0]*0.0)
#i_stop = h_data.shape[0]

i_span = i_stop - i_start
t = np.zeros(i_span, np.float32)
x0 = np.zeros(i_span, np.float32)
y0 = np.zeros(i_span, np.float32)
h0 = np.zeros(i_span, np.float32)
yaw0 = np.zeros(i_span, np.float32)

for i in range(i_start,i_stop):
    
    i_now = i - i_start
    t[i_now] = i_now * dt

    print(np.round((i_now+1)/i_span*100,2),"% Complete")
    
    # Extract Data
    h_new = np.reshape(h_data[i,:], (imax,jmax))
    mpc_new = mpc_data[i,:]
    for j in range(N+1):
        x[j] = mpc_new[3*j+0]
        y[j] = mpc_new[3*j+1]
        yaw[j] = mpc_new[3*j+2]
        u[j] = 1.0 * np.cos(yaw[j])
        v[j] = 1.0 * np.sin(yaw[j])
    xd = mpc_new[3*N + 3*(N+1) + 0]
    yd = mpc_new[3*N + 3*(N+1) + 1]

    # Current State
    x0[i_now] = x[0]
    y0[i_now] = y[0]
    yaw0[i_now] = yaw[0]

    # Fractional Index Corresponding to Current Position
    ir = (imax-1) - y0[i_now] / ds
    jr = x0[i_now] / ds

    # Bilinear Interpolation
    i1f = np.floor(ir)
    j1f = np.floor(jr)
    i2f = np.ceil(ir)
    j2f = np.ceil(jr)

    i1 = int(i1f)
    j1 = int(j1f)
    i2 = int(i2f)
    j2 = int(j2f)

    if (i1 != i2) and (j1 != j2):
        f1 = (i2f - ir) * h_new[i1,j1] + (ir - i1f) * h_new[i2,j1]
        f2 = (i2f - ir) * h_new[i1,j2] + (ir - i1f) * h_new[i2,j2]
        h0[i_now] = (j2f - jr) * f1 + (jr - j1f) * f2
    elif (i1 != i2):
        h0[i_now] = (i2f - ir) * h_new[i1,int(jr)] + (ir - i1f) * h_new[i2,int(jr)]
    elif (j1 != j2):
        h0[i_now] = (j2f - jr) * h_new[int(ir),j1] + (jr - j1f) * h_new[int(ir),j2]
    else:
        h0[i_now] = h_new[int(ir),int(jr)]

    # Plot
    ax1[0,0].clear()
    ax1[0,1].clear()
    ax1[0,2].clear()
    surf1 = ax1[0,0].pcolormesh(X, (imax-1)*ds-Y, h_new, vmin = -0.1, vmax = 0.7, cmap=style)
    cont1 = ax1[0,0].contour(X, (imax-1)*ds-Y, h_new, levels=[0.001], linewidths={3}, cmap=style)
    traj1 = ax1[0,0].quiver(x,y,u,v)
    traj2 = ax1[0,0].scatter(x[0],y[0], color='black', marker='s', s=100)
    traj3 = ax1[0,0].scatter(xd, yd, color='green', marker='*', s=250)
    ax1[0,0].set_xlim(0,(imax-1)*ds)
    ax1[0,0].set_ylim(0,(imax-1)*ds)
    ax1[0,0].set_aspect("equal")

    ax1[0,1].plot(t[:i_now],h0[:i_now])
    ax1[0,1].plot(t[:i_now],0.0*h0[:i_now])
    ax1[0,1].set_xlabel("time (s)")
    ax1[0,1].set_ylabel("h")
    ax1[0,1].set_xlim(0,i_span*dt)
    ax1[0,2].plot(t[:i_now],yaw0[:i_now]*180.0/np.pi)
    ax1[0,2].set_xlabel("time (s)")
    ax1[0,2].set_ylabel("yaw")
    ax1[0,2].set_xlim(0,i_span*dt)

    if i == (i_stop-1):
        plt.show()
    else:
        plt.waitforbuttonpress(0.001)
    