import sys
import numpy as np
from scipy import optimize
import pylab as pl

nPoints = 40                  # discretization
# number of springs is nPoints - 1
# number of masses is nPoints

k = 30.0
w = 37.0
L = 1.0
D = 0.2

qk = k * (nPoints - 1)                # N/m
qw = w / nPoints                      # N
qEquilibriumLength = L / (nPoints - 1) # m

xStart = np.sin(np.linspace(0, 2.0 * np.pi, nPoints))
yStart = np.linspace(0, D, nPoints)
zStart = np.zeros(nPoints)

xyzs = np.c_[xStart, yStart, zStart]
xyzSift = np.array([[D/2.0, 0.0, 0.0]])
# print 'coordinates shape', xyzs.shape # gives (nPoints, 3)
# print 'coordinates shape', xyzSift.shape # gives (1, 3)


def closenessPenalty(dist, eqDist=qEquilibriumLength):
    penalty = (eqDist - dist)**3 
    penalty[dist > eqDist] = 0
    return penalty

def selfInteraction(xyzCoordinates, xyzSift):
    xyzs = xyzCoordinates.reshape((nPoints, 3)).T
    xs1, ys1, zs1 = xyzs
    xs2, ys2, zs2 = xyzs - xyzSift.T
    
    dxsq = (xs1[:,np.newaxis] - xs2[np.newaxis,:]) ** 2
    dysq = (ys1[:,np.newaxis] - ys2[np.newaxis,:]) ** 2
    dzsq = (zs1[:,np.newaxis] - zs2[np.newaxis,:]) ** 2

    dL = np.sqrt(dxsq + dysq + dzsq)

    return closenessPenalty(dL).sum()

def computeEnergy(xyzCoordinates):
    '''Given a (n, 3) vector of points.  This functions
    computes the total energy of the configuration'''

    # Unpack the xs, ys and zs
    xs, ys, zs = xyzCoordinates.reshape((nPoints, 3)).T

    # Add some constraints / boundary conditions
    xs[0] = 0
    xs[-1] = 0
    ys[0] = 0
    ys[-1] = D
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

    energeticContent = totalSpringEnergy +  selfInteraction(xyzCoordinates, xyzSift)
    # TODO: Remove gravity and add self interactions

    print "\rEnergy:%f" % energeticContent,
    sys.stdout.flush()

    return energeticContent


# Optimization of X, Y, and Z
minimumXYZs = optimize.fmin_cg(computeEnergy, xyzs, maxiter=100)

# print 'minimumXYZs.shape = ', minimumXYZs.shape # gives (3 * nPoints, )
xs, ys, zs = minimumXYZs.reshape((nPoints, 3)).T

pl.figure()
pl.plot(xs, ys, "ro", alpha=0.7)
pl.plot(xStart, yStart, "bo", alpha=0.7)
pl.axis("equal")
pl.xlabel('x')
pl.ylabel('y')

# pl.figure()
# pl.plot(xs, zs, "ro", alpha=0.7)
# pl.plot(xs, zs, "b:", alpha=0.7)
# pl.axis("equal")
# pl.xlabel('x')
# pl.ylabel('z')

# pl.figure()
# pl.plot(np.diff(xs), "ro", alpha=0.7)
# pl.xlabel('dx')

# pl.figure()
# pl.plot(np.diff(ys), "ro", alpha=0.7)
# pl.xlabel('dy')

# pl.figure()
# pl.plot(np.diff(zs), "ro", alpha=0.7)
# pl.xlabel('dz')

# print "DX"
# print np.diff(xs)

# print "DY"
# print np.diff(ys)


pl.show()
