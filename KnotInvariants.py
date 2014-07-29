import numpy as np
from mayavi.mlab import *
from scipy import integrate, interpolate, linalg

def estNormedGradient(func, point, lambder = 1e-8):
    dF = (func(point + lambder / 2.0) - func(point - lambder / 2.0)) / lambder
    dF /= linalg.norm(dF)
    return dF

def triplify(aray):
    return np.concatenate((aray, aray, aray))

def crossDot(v1, v2, v3):
    return np.dot(v1, np.cross(v2, v3))

twoPi = np.pi * 2

def computeLinkingNumber(pointSet1, pointSet2):
    # Make a spline of each of the input point sets, with periodic wrapping
    pts1 = np.concatenate((pointSet1, pointSet1))
    tht1 = np.linspace(-twoPi, twoPi, pts1.shape[0])
    sp1 = interpolate.interp1d(tht1, pts1.T, kind="cubic")

    # once = np.linspace(-np.pi, np.pi, 100)
    # circle = sp1(once)
    # plot3d(circle[0, :], circle[1, :], circle[2, :])

    # grad = np.array([estNormedGradient(sp1, tht) for tht in once]).T
    # quiver3d(circle[0, :], circle[1, :], circle[2, :], 
    #          grad[0, :], grad[1, :], grad[2, :]) 
    # show()

    pts2 = np.concatenate((pointSet2, pointSet2))
    tht2 = np.linspace(-twoPi, twoPi, pts2.shape[0])
    sp2 = interpolate.interp1d(tht2, pts2.T, kind="cubic")

    def integrateMe(phi1, phi2):
        g1 = estNormedGradient(sp1, phi1)
        g2 = estNormedGradient(sp2, phi2)

        dif = sp2(phi2) - sp1(phi1)
        dist3 = linalg.norm(dif)**3
        link = crossDot(dif, g1, g2) / (np.pi * dist3 * 4.0)

        return link

    return integrate.dblquad(integrateMe, -np.pi, np.pi, 
                             lambda x: -np.pi, lambda y: np.pi)


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

    # plot3d(xs1, ys1, zs1, color=(0.0, 0.8, 0.8), tube_radius=0.1)
    # plot3d(xs2, ys2, zs2, color=(0.8, 0.8, 0.0), tube_radius=0.1)
    # savefig('circle.png')
    # show()

    print computeLinkingNumber(xyz1, xyz2)
