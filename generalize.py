import sys
import numpy as np
from scipy import optimize
from mayavi.mlab import *

nPoints = 100                  # discretization
# number of springs is nPoints - 1
# number of masses is nPoints

k = 0.04
L = 1.0
radius = 0.08

qk = k * (nPoints - 1)                 # N/m
qEquilibriumLength = L / (nPoints - 1) # m


xHalf = 0.3 * np.sin(np.linspace(0, 2.0 * np.pi, nPoints)) + np.linspace(0, L, nPoints)
yHalf = np.linspace(0, L, nPoints)
zHalf = np.cos(np.linspace(0, 2.0 * np.pi, nPoints)) - 1.0

xStart = np.concatenate((-xHalf[::-1], xHalf[1:]))
yStart = np.concatenate((yHalf[::-1], yHalf[1:]))
zStart = np.concatenate((zHalf[::-1], zHalf[1:]))

xyzs = np.c_[xStart, yStart, zStart]
xyzShift = np.array([[0.0, L / 2.0, 0.0]])

def Reshape3Stiches(xyzCoordinates):
    xyzs = xyzCoordinates.reshape((2 * nPoints - 1, 3)).T
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

    return closenessPenalty(dLup).sum() + closenessPenalty(dLdown).sum()

def computeEnergy(xyzCoordinates):
    '''Given a (n, 3) vector of points.  This functions
    computes the total energy of the configuration'''

    # Unpack the xs, ys and zs
    xs, ys, zs = xyzCoordinates.reshape((2 * nPoints - 1, 3)).T

    # Add some constraints / boundary conditions
    xs[0], xs[-1] = -L, L
    # ys[1], ys[-1] = ys[0], ys[-2]
    # zs[1], zs[-1] = zs[0], zs[-2] 

    # flatten again
    xyzCoordinates[:] =  np.c_[xs, ys, zs].flatten()

    # Spring Energy Computation
    dx = np.diff(xs)
    dy = np.diff(ys)
    dz = np.diff(zs)
    dL = np.sqrt( (dx**2) + (dy**2) + (dz**2) )

    displacementFromEq = dL - qEquilibriumLength
    springEnergy = qk * displacementFromEq**2 / 2.0
    totalSpringEnergy = springEnergy.sum()
    energeticContent = totalSpringEnergy + selfInteraction(xyzCoordinates)

    print "\rEnergy: %5.5f           " % energeticContent,
    sys.stdout.flush()

    return energeticContent

# Optimize and plot
colors = np.linspace(0.0, 1.0, 5)
for c in colors:
    # Optimization of X, Y, and Zs
    xyzs = optimize.fmin_cg(computeEnergy, xyzs, maxiter = 10, epsilon = 0.001)
    
    # Plot 3 stiches
    xs1, ys1, zs1, xs2, ys2, zs2, xs3, ys3, zs3 = Reshape3Stiches(xyzs)
    plot3d(ys1, zs1, xs1, color=(0.0, c, c), tube_radius=radius)
    plot3d(ys2, zs2, xs2, color=(0.0, c, c), tube_radius=radius)
    plot3d(ys3, zs3, xs3, color=(0.0, c, c), tube_radius=radius)
show()
