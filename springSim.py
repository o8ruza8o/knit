import sys
import numpy as np

nPoints = 100                  # discretization
# number of springs is nPoints - 1
# number of masses is nPoints

k = 10.0
w = 37.0

qk = k * (nPoints - 1)                # N/m
qw = w / nPoints                      # N
equilibriumLength = 1.0 / (nPoints - 1) # m

xPositions = np.zeros(nPoints)
yPositions = np.linspace(0, -2, nPoints)

xys = np.c_[xPositions, yPositions]

print xys.shape


def computeEnergy(xyCoordinates):
    print xyCoordinates.shape
    # Unpack the xs and ys
    xs, ys = xyCoordinates.reshape((nPoints, 2)).T

    # Constrain Problem to only changes in y position
    xs[:] = 0
    ys[0] = 0

    # Spring Energy Computation
    dx = np.diff(xs)                      # m
    dy = np.diff(ys)                      # m
    dL = np.sqrt( (dx**2) + (dy**2) )     # m

    displacementFromEq = dL - equilibriumLength   # m
    springEnergy = qk * displacementFromEq**2
    
    totalSpringEnergy = springEnergy.sum()

    # "Gravatational" energy (wrt (0,0))
    gravEnergy = qw * ys

    totalGravEnergy = gravEnergy.sum()

    energeticContent = totalSpringEnergy + totalGravEnergy
    print totalSpringEnergy, totalGravEnergy, energeticContent, ys.min()

    return energeticContent

def computeEnergy1D(ys):
    # Constrain Problem to only changes in y position
    ys[0] = 0

    # Spring Energy Computation
    dy = np.sqrt(np.diff(ys)**2)

    displacementFromEq = dy - equilibriumLength
    springEnergy = qk * displacementFromEq**2 / 2.0

    totalSpringEnergy = springEnergy.sum()

    # "Gravatational" energy (wrt (0,0))
    gravEnergy = qw * ys

    totalGravEnergy = gravEnergy.sum()

    energeticContent = totalSpringEnergy + totalGravEnergy
    # print "\r Sp:", totalSpringEnergy,
    # sys.stdout.flush()
    # print "Gr:", totalGravEnergy
    # print "Et:", energeticContent
    print "\r\t\t dy:", (ys.max() - ys.min()) - 1.0,
    sys.stdout.flush()

    return energeticContent


from scipy import optimize
# Optimization of X and Y
# minimumXYs = optimize.fmin(computeEnergy, xys)   #


minimumYs = optimize.fmin_cg(computeEnergy1D, yPositions) #, ftol=1e-8, xtol=1e-6, maxiter=1e8, maxfun = 1e8)

xs = np.zeros_like(minimumYs)
ys = minimumYs

from pylab import axis, plot, show, figure

plot(xs, ys, "ro", alpha=0.7)
plot(xs, ys, "b:", alpha=0.7)
axis("equal")

figure()
plot(np.diff(ys), "ro", alpha=0.7)
axis("equal")

print "\t\t w/k:", w / (2.0 * k)

# print "DX"
# print np.diff(xs)

# print "DY"
# print np.diff(ys)


show()
