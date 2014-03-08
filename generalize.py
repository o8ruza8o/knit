import sys
import numpy as np
from scipy import optimize
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mayavi.mlab import *

nPoints = 100                  # discretization
# number of springs is nPoints - 1
# number of masses is nPoints

k = 0.4
w = 37.0
L = 1.0
D = 0.2

qk = k * (nPoints - 1)                # N/m
qw = w / nPoints                      # N
qEquilibriumLength = L / (nPoints - 1) # m

xStart = 0.3 * np.sin(np.linspace(0, 2.0 * np.pi, nPoints)) + np.linspace(0, L, nPoints)
yStart = np.linspace(0, L, nPoints)
zStart = np.cos(np.linspace(0, 2.0 * np.pi, nPoints)) - 1.0

xyzs = np.c_[xStart, yStart, zStart]
xyzShift = np.array([[0.0, L / 2.0, 0.0]])
# print 'coordinates shape', xyzs.shape # gives (nPoints, 3)
# print 'coordinates shape', xyzShift.shape # gives (1, 3)

def closenessPenalty(dist):
    penalty = (L / 2.0 - dist)**2 
    penalty[dist > L / 2.0] = 0
    return penalty / nPoints

def selfInteraction(xyzCoordinates):
    xyzs = xyzCoordinates.reshape((nPoints, 3)).T
    xs1, ys1, zs1 = xyzs - xyzShift.T
    xs2, ys2, zs2 = xyzs
    xs3, ys3, zs3 = xyzs + xyzShift.T
    
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
    xs, ys, zs = xyzCoordinates.reshape((nPoints, 3)).T

    # Add some constraints / boundary conditions
    xs[0], xs[-1] = 0.0, L
    ys[1], ys[-1] = ys[0], ys[-2]
    zs[1], zs[-1] = zs[0], zs[-2] 

    xyzCoordinates[:] =  np.c_[xs, ys, zs].flatten()

    # Spring Energy Computation
    dx = np.diff(xs)
    dy = np.diff(ys)
    dz = np.diff(zs)
    dL = np.sqrt( (dx**2) + (dy**2) + (dz**2) )

    displacementFromEq = dL - qEquilibriumLength
    springEnergy = qk * displacementFromEq**2 / 2.0
    totalSpringEnergy = springEnergy.sum()

    # "Gravatational" Energy Computation (wrt (0,0))
    # gravEnergy = qw * ys
    # totalGravEnergy = gravEnergy.sum()

    energeticContent = totalSpringEnergy + selfInteraction(xyzCoordinates)

    print "\rEnergy: %5.5f           " % energeticContent,
    sys.stdout.flush()

    return energeticContent

# Create a figure
colors = np.linspace(0.0, 1.0, 3)
for c in colors:
    # Reshape
    xs, ys, zs = xyzs.reshape((nPoints, 3)).T
    x2 = np.concatenate((-xs[::-1], xs[1:]))
    y2 = np.concatenate((ys[::-1], ys[1:]))
    z2 = np.concatenate((zs[::-1], zs[1:]))

    # What is going on with BCs
    print "Xs ", xs[0], xs[-1]
    print "Ys ", ys[1], ys[-1], ys[0], ys[-2]
    print "Zs ", zs[1], zs[-1], zs[0], zs[-2] 


    xyz = np.c_[x2, y2, z2]
    x1, y1, z1 = (xyz - xyzShift).T
    x3, y3, z3 = (xyz + xyzShift).T

    # Plot first
    plot3d(y1, z1, x1, color=(0.0, c, c), tube_radius=0.025)
    plot3d(y2, z2, x2, color=(0.0, c, c), tube_radius=0.025)
    plot3d(y3, z3, x3, color=(0.0, c, c), tube_radius=0.025)

    # Optimization of X, Y, and Z
    xyzs = optimize.fmin_cg(computeEnergy, xyzs, maxiter = 10, epsilon = 0.001)
show()
