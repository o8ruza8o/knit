import sys
import numpy as np
from scipy import optimize
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

nPoints = 40                  # discretization
# number of springs is nPoints - 1
# number of masses is nPoints

k = 300.0
w = 37.0
L = 1.0
D = 0.2

qk = k * (nPoints - 1)                # N/m
qw = w / nPoints                      # N
qEquilibriumLength = L / (nPoints - 1) # m

xStart = np.sin(np.linspace(0, 2.0 * np.pi, nPoints))
yStart = np.linspace(0, L, nPoints)
zStart = np.sin(np.linspace(0, 2.0 * np.pi, nPoints))

xyzs = np.c_[xStart, yStart, zStart]
xyzShift = np.array([[0.0, L / 4.0, 0.0]])
# print 'coordinates shape', xyzs.shape # gives (nPoints, 3)
# print 'coordinates shape', xyzShift.shape # gives (1, 3)

def closenessPenalty(dist, eqDist=qEquilibriumLength):
    penalty = (eqDist - dist)**3 
    penalty[dist > eqDist] = 0
    return penalty

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
    xs[0] = 0
    xs[-1] = 0
    ys[0] = 0
    ys[-1] = L
    zs[0] = 0
    zs[-1] = 0

    # Spring Energy Computation
    dx = np.diff(xs)                      # m
    dy = np.diff(ys)                      # m
    dz = np.diff(zs)                      # m
    dL = np.sqrt( (dx**2) + (dy**2) + (dz**2) )     # m

    displacementFromEq = dL - qEquilibriumLength   # m
    springEnergy = qk * displacementFromEq**2 / 2.0
    totalSpringEnergy = springEnergy.sum()

    # "Gravatational" Energy Computation (wrt (0,0))
    # gravEnergy = qw * ys
    # totalGravEnergy = gravEnergy.sum()

    energeticContent = totalSpringEnergy +  selfInteraction(xyzCoordinates)

    print "\rEnergy: %5.5f           " % energeticContent,
    sys.stdout.flush()

    return energeticContent

# 3d plot
fig = plt.figure()
ax = fig.gca(projection='3d')
colors = np.linspace(0.0, 1.0, 10)
for c in colors:
    xs, ys, zs = xyzs.reshape((nPoints, 3)).T
    ax.plot(xs, ys, zs, color=(0.0, c, c), alpha=0.7, lw=1, ls='-', marker='o', markersize=8)
    # Optimization of X, Y, and Z
    xyzs = optimize.fmin_cg(computeEnergy, xyzs, maxiter = 10)

ax.axis("equal")
plt.show()
