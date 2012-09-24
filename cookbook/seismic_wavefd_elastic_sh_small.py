"""
Seis: 2D finite difference simulation of elastic SH wave propagation
"""
import time
import numpy as np
import fatiando as ft

log = ft.log.get()

# Make a wave source from a mexican hat wavelet
sources = [ft.seis.wavefd.MexHatSource(25, 25, 100, 0.5, delay=1.5)]
# Set the parameters of the finite difference grid
shape = (50, 50)
spacing = (1000, 1000)
area = (0, spacing[1]*shape[1], 0, spacing[0]*shape[0])
# Make a density and S wave velocity model
dens = 2700*np.ones(shape)
svel = 3000*np.ones(shape)

# Get the iterator. This part only generates an iterator object. The actual
# computations take place at each iteration in the for loop bellow
dt = 0.05
maxit = 300
timesteps = ft.seis.wavefd.elastic_sh(spacing, shape, svel, dens, dt, maxit,
    sources, padding=0.5)

vmin, vmax = -1*10**(-4), 1*10**(-4)
# This part makes an animation by updating the plot every few iterations
ft.vis.ion()
ft.vis.figure()
ft.vis.axis('scaled')
x, z = ft.grd.regular(area, shape)
# Start with everything zero and grab the plot so that it can be updated later
wavefield = ft.vis.pcolor(x, z, np.zeros(shape).ravel(), shape, vmin=vmin,
    vmax=vmax)
# Make z positive down
ft.vis.ylim(area[-1], area[-2])
ft.vis.m2km()
ft.vis.xlabel("x (km)")
ft.vis.ylabel("z (km)")
start = time.clock()
for i, u in enumerate(timesteps):
    if i%10 == 0:
        ft.vis.title('time: %0.1f s' % (i*dt))
        wavefield.set_array(u[0:-1,0:-1].ravel())
        ft.vis.draw()
ft.vis.ioff()
print 'Frames per second (FPS):', float(i)/(time.clock() - start)
ft.vis.show()