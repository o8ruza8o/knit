import sys
import numpy as np
from scipy import optimize
from mayavi.mlab import *

nPoints = 100               # discretization
# number of springs is nPoints - 1
# number of masses is nPoints

k = 0.01
L = 1.0
radius = 0.15

qk = k * (nPoints - 1)                 # N/m
qEquilibriumLength = L / (nPoints - 1) # m


xHalf = 0.3 * np.sin(np.linspace(0, 2.0 * np.pi, nPoints)) + np.linspace(0, L, nPoints)
yHalf = np.linspace(0, L, nPoints)
zHalf = np.cos(np.linspace(0, 2.0 * np.pi, nPoints)) - 1.0

xStart = np.concatenate((-xHalf[::-1], xHalf[:]))
yStart = np.concatenate((yHalf[::-1], yHalf[:]))
zStart = np.concatenate((zHalf[::-1], zHalf[:]))

xyzs = np.c_[xStart, yStart, zStart]
xyzShift = np.array([[0.0, L / 2.0, 0.0]])

def Reshape3Stiches(xyzCoordinates):
    xyzs = xyzCoordinates.reshape((2 * nPoints, 3)).T
    xs1, ys1, zs1 = xyzs - xyzShift.T
    xs2, ys2, zs2 = xyzs
    xs3, ys3, zs3 = xyzs + xyzShift.T
    return xs1, ys1, zs1, xs2, ys2, zs2, xs3, ys3, zs3

def closenessPenalty(dist):
    penalty = (2.0 * radius - dist)**2 
    penalty[dist > 2.0 * radius] = 0
    return penalty / nPoints

def selfInteraction(xyzCoordinates):
    xs1, ys1, zs1, xs2, ys2, zs2, xs3, ys3, zs3 = Reshape3Stiches(xyzCoordinates)

    dxsq = (xs1[:,np.newaxis] - xs2[np.newaxis,:]) ** 2
    dysq = (ys1[:,np.newaxis] - ys2[np.newaxis,:]) ** 2
    dzsq = (zs1[:,np.newaxis] - zs2[np.newaxis,:]) ** 2

    dLdown = np.sqrt(dxsq + dysq + dzsq)

    dxsq = (xs2[:,np.newaxis] - xs3[np.newaxis,:]) ** 2
    dysq = (ys2[:,np.newaxis] - ys3[np.newaxis,:]) ** 2
    dzsq = (zs2[:,np.newaxis] - zs3[np.newaxis,:]) ** 2

    dLup = np.sqrt(dxsq + dysq + dzsq)

    dxsq = (xs1[:,np.newaxis] - xs3[np.newaxis,:]) ** 2
    dysq = (ys1[:,np.newaxis] - ys3[np.newaxis,:]) ** 2
    dzsq = (zs1[:,np.newaxis] - zs3[np.newaxis,:]) ** 2

    dLcross = np.sqrt(dxsq + dysq + dzsq)

    return ( closenessPenalty(dLup).sum() + 
             closenessPenalty(dLdown).sum() +
             closenessPenalty(dLcross).sum())

def computeEnergy(xyzCoordinates):
    '''Given a (n, 3) vector of points.  This functions
    computes the total energy of the configuration'''

    # Unpack the xs, ys and zs
    xs, ys, zs = xyzCoordinates.reshape((2 * nPoints, 3)).T

    # Add some constraints / boundary conditions
    xs[0], xs[-1] = -L, L
    # ys[1], ys[-1] = ys[0], ys[-2]
    # zs[1], zs[-1] = zs[0], zs[-2] 

    # flatten again
    xyzCoordinates[:] =  np.c_[xs, ys, zs].flatten()

    # Spring Energy Computation
    dx =  (np.diff(xs) + 
           np.concatenate(([np.diff(xs)[-1]], np.diff(xs)[:-1]))) / 2.0
    dy = (np.diff(ys) + 
           np.concatenate(([np.diff(ys)[-1]], np.diff(ys)[:-1]))) / 2.0
    dz = (np.diff(zs) + 
           np.concatenate(([np.diff(zs)[-1]], np.diff(zs)[:-1]))) / 2.0
    dL = np.sqrt( (dx**2) + (dy**2) + (dz**2) )

    displacementFromEq = dL - qEquilibriumLength
    springEnergy = qk * displacementFromEq**2 / 2.0
    totalSpringEnergy = springEnergy.sum()
    energeticContent = totalSpringEnergy + selfInteraction(xyzCoordinates)

    print "\rEnergy: %5.5f           " % energeticContent,
    sys.stdout.flush()

    return energeticContent

def mayaviscene():
    scene.scene.y_minus_view()

    scene.scene.camera.position = [0.5, -4.7, 0.0]
    scene.scene.camera.focal_point = [0.5, -1.0, 0.0]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [1.0, 0.0, 0.0]
    scene.scene.camera.clipping_range = [1.4, 6.6]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

# Optimize and plot
colors = np.linspace(0.0, 1.0, 6)
for c in colors:
    # Optimization of X, Y, and Zs
    print "\n  percent done  " + "{0:.2f}".format(c)
    xyzs = optimize.fmin_cg(computeEnergy, xyzs, maxiter = 10, epsilon = 0.001)

# Plot 3 stiches
scene = figure()
xs1, ys1, zs1, xs2, ys2, zs2, xs3, ys3, zs3 = Reshape3Stiches(xyzs)
plot3d(ys1[1::2], zs1[1::2], xs1[1::2], color=(0.8*c, 0.8*c, 0.0), tube_radius=radius)
plot3d(ys2[1::2], zs2[1::2], xs2[1::2], color=(c, c, 0.0), tube_radius=radius)
plot3d(ys3[1::2], zs3[1::2], xs3[1::2], color=(0.8*c, 0.8*c, 0.0), tube_radius=radius)
mayaviscene()
savefig('stich.obj')
savefig('stich.png')
show()

