import random, sys
import numpy as np
from scipy import optimize

from pylab import axis, plot, show, figure, title

G = np.array((0.0,0.0, 1.0)).reshape((3,1))

class Segment(object):
    def __init__(self, xyz, totalMass=10.0, totalSpringCoeff=1.0):
        assert xyz.shape[0] == 3, "Format"
        assert xyz.ndim == 2, "Format"

        nPoints = xyz.shape[1]

        self.xyz = xyz
        self.eqLength = self.computeSpringLengthDiffs()

        self.qK = totalSpringCoeff * (nPoints - 1)   # N/m
        self.qW = totalMass / nPoints                # N

    def computeSpringLengthDiffs(self):
        dxyz_sq = np.diff(self.xyz, axis=1)**2
        return np.sqrt(dxyz_sq.sum(axis=0))

    def computeGravatitionalEnergy(self, gravityVector=G):
        gVec = self.xyz * gravityVector * self.qW
        return gVec.sum()

    def computeElasticEnergy(self):
        # Spring Energy Computation
        dL = self.computeSpringLengthDiffs()

        displacement = self.eqLength - dL
        dEVec = self.qK * displacement**2
        return sum(dEVec)

    def closenessPenalty(dist, eqDist=None):
        penalty = (eqDist - dist)**3
        penalty[dist > eqDist] = 0
        return penalty

    def computeInteferenceEnergy(self, interfereXYZ):
        xs1, ys1, zs1 = self.xyz
        xs2, ys2, zs2 = interfereXYZ

        dxsq = (xs1[:,np.newaxis] - xs2[np.newaxis,:]) ** 2
        dysq = (ys1[:,np.newaxis] - ys2[np.newaxis,:]) ** 2
        dzsq = (zs1[:,np.newaxis] - zs2[np.newaxis,:]) ** 2

        dL = np.sqrt(dxsq + dysq + dzsq)

        # Ignore self-self particle interactions
        # aka treat as inf far away
        if interfereXYZ is self.xyz:
            wipe = arange(self.xyz.shape[1])
            dL[wipe, wipe] = np.inf

        return closenessPenalty(dL).sum()

    def plotXZ(self, color=(0,0,1)):
        xs = self.xyz[0,:]
        zs = self.xyz[2,:]
        plot(xs, zs, ".", color=color, alpha=0.7)
        plot(xs, zs, ":", color=color, alpha=0.7)
        axis("equal")

    def plotYZ(self, color=(0,0,1)):
        ys = self.xyz[1,:]
        zs = self.xyz[2,:]
        plot(ys, zs, ".", color=color, alpha=0.7)
        plot(ys, zs, ":", color=color, alpha=0.7)
        axis("equal")

    def BCEnergyContribution(self):
        pass

    def getDOF(self):
        return self.xyz.flatten()

    def setDOF(self, values):
        self.xyz[:] = values.reshape((3,-1))


class FixedPointSegment(Segment):
    def getDOF(self):
        return self.xyz[:,1:].flatten().copy()

    def setDOF(self, values):
        self.xyz[:,1:][:] = values.reshape((3,-1))


class FixedEndsSegment(Segment):
    def getDOF(self):
        return self.xyz[:,1:-1].flatten().copy()

    def setDOF(self, values):
        self.xyz[:,1:-1][:] = values.reshape((3,-1))


class Simulator(object):
    def __init__(self, segmentList):
        self.segments = segmentList
        self.DOF = sum([segment.getDOF().size for segment in self.segments])

    def assignAllDOF(self, values):
        assert values.size == self.DOF, "Wrong size!" + str(values.shape) + " " + str(self.DOF)

        offset = 0
        for segment in self.segments:
            currentSegmentSize = segment.getDOF().size
            toSet = values[offset:offset+currentSegmentSize]
            segment.setDOF(toSet)
            offset += currentSegmentSize

    def retrieveAllDOF(self):
        return np.concatenate([segment.getDOF().flatten() for segment in self.segments])

    def plotAllSegments(self, color=(0,0,1)):
        figure(1)
        title("XZ")
        figure(2)
        title("YZ")

        for n, segment in enumerate(self.segments):
            color = ( (len(self.segments) - 1.0 * n) / len(self.segments), 0.0,  1.0 * n / len(self.segments))
            figure(1)
            segment.plotXZ(color=color)
            figure(2)
            segment.plotYZ(color=color)


    def run(self):
        startingDOFs = self.retrieveAllDOF()
        print startingDOFs.shape
        result = optimize.fmin_bfgs(self.energyFunction, startingDOFs)
        # result = optimize.fmin(self.energyFunction, result)
        self.assignAllDOF(result)

class GravatationalSpringSimulator(Simulator):
    def energiesToGather(self):
        return ["computeGravatitionalEnergy", "computeElasticEnergy"]
    def energyFunction(self, allDOF, printPercent = 5 ):
        self.assignAllDOF(allDOF)

        energies = {}
        for segment in self.segments:
            for energyName in self.energiesToGather():
                energies[energyName] = energies.get(energyName, 0) + getattr(segment, energyName)()

        energyTotal = sum(energies.values())

        if random.uniform(0.0, 100.0) < printPercent:
            print "\r", " ".join(["%s: %0.09f" % eTup for eTup in energies.items()]), "\t\t", energyTotal,
            sys.stdout.flush()

        return  energyTotal


def segmentifyParamaterized(fx, fy, fz, tStart, tEnd, tSteps):
    t = np.linspace(tStart, tEnd, tSteps)
    return np.c_[fx(t), fy(t), fz(t)]


def testHangingSegment(w, k):
    segs = []

    nPoints = 10 * (n + 1)
    xyz = np.c_[np.zeros(nPoints) + n,
                np.zeros(nPoints),
                np.linspace(0.0, -1.0, nPoints)].T

    fps = FixedPointSegment(xyz, totalSpringCoeff=10000)

    segs.append(fps)

    sim = GravatationalSpringSimulator(segs)
    sim.run()
    sim.plotAllSegments()


def testCatenarySegment(w, k):
    segs = []

    for x in range(5):
        nPoints = 20
        xyz = np.c_[np.linspace(1.0, -1.0, nPoints),
                    np.zeros(nPoints),
                    np.zeros(nPoints)].T

        fps = FixedEndsSegment(xyz,totalSpringCoeff=10**x)
        segs.append(fps)

    sim = GravatationalSpringSimulator(segs)
    sim.run()
    sim.plotAllSegments()


testCatenarySegment(0,0)
