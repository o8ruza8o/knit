import numpy as np
from scipy import integrate, interpolate, linalg

def estNormedGradient(func, point, lambder = 1e-8):
    dF = (func(point) - func(point + lambder)) / lambder
    dF /= linalg.norm(dF)
    return dF

def triplify(aray):
    return np.concatenate((aray, aray, aray))

def crossDot(v1, v2):
    return np.dot(np.cross(v1, v2), v2)

twoPi = np.pi * 2

def computeLinkingNumber(pointSet1, pointSet2):
    # Make a spline of each of the input point sets, with periodic wrapping
    pts1 = np.concatenate((pointSet1, pointSet1))
    tht1 = np.linspace(-twoPi, twoPi, pts1.shape[0])
    sp1 = interpolate.interp1d(tht1, pts1.T, kind="quadratic")


    pts2 = np.concatenate((pointSet2, pointSet2))
    tht2 = np.linspace(-twoPi, twoPi, pts2.shape[0])
    sp2 = interpolate.interp1d(tht2, pts2.T, kind="quadratic")

    def integrateMe(phi1, phi2):
        g1 = estNormedGradient(sp1, phi1)
        g2 = estNormedGradient(sp2, phi2)

        dist = linalg.norm(g2 - g1)
        cd = crossDot(g1, g2)

        return (dist**3/dist) * cd

    return integrate.dblquad(integrateMe, 0, twoPi, lambda x: 0, lambda y: twoPi)


if __name__ == "__main__":
    p = np.linspace(0, twoPi, 100)

    r1 = 2
    xs1 = r1 * np.cos(p)
    ys1 = r1 * np.sin(p)
    zs1 = np.ones_like(p)
    xyz1 = np.c_[xs1, ys1, zs1]


    r2 = 2
    xs2 = r2 * np.cos(p) + 1
    ys2 = np.ones_like(p) * 0
    zs2 = r2 * np.sin(p)
    xyz2 = np.c_[xs2, ys2, zs2]

    print computeLinkingNumber(xyz1, xyz2)

