import numpy as np
from mayavi.mlab import *
from scipy import integrate, interpolate, linalg

def estNormedGradient(func, point, epsilon= 1e-5):
    dF = (func(point + epsilon / 2.0) - func(point - epsilon / 2.0)) / epsilon
    return dF

def dotCross(v1, v2, v3):
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

        dist = sp1(phi1) - sp2(phi2)
        dist3 = linalg.norm(dist)**3
        link = dotCross(dist, g1, g2) / (np.pi * dist3 * 4.0)

        return link

    return integrate.dblquad(integrateMe, -np.pi, np.pi, 
                             lambda x: -np.pi, lambda y: np.pi, 
                             epsabs = 1e-3, epsrel = 1e-3)


if __name__ == "__main__":
    # print integrate.dblquad(lambda x, y: x, 0, 1, lambda x: 0, lambda y: 5)

    p = np.linspace(0, twoPi, 100)

    # r1 = 2
    # xs1 = r1 * np.cos(p)
    # ys1 = r1 * np.sin(p)
    # zs1 = np.ones_like(p) * 0
    # xyz1 = np.c_[xs1, ys1, zs1]

    # r2 = 2
    # xs2 = r2 * np.cos(p) + 1
    # ys2 = np.ones_like(p) * 0
    # zs2 = r2 * np.sin(p)
    # xyz2 = np.c_[xs2, ys2, zs2]

    # plot3d(xs1, ys1, zs1, color=(0.0, 0.8, 0.8), tube_radius=0.1)
    # plot3d(xs2, ys2, zs2, color=(0.8, 0.8, 0.0), tube_radius=0.1)
    # savefig('knot.png')
    # show()

    a = 1.8
    x0 = 0.1
    y0 = 0.1
    z0 = -0.1

    xs1 = (a +  np.cos(5 * p)) * np.cos(2 * p)
    ys1 = (a +  np.cos(5 * p)) * np.sin(2 * p)
    zs1 = np.sin(5 * p)
    xyz1 = np.c_[xs1, ys1, zs1]

    xs2 = x0 - a * np.cos(p)
    ys2 = y0 + a * np.sin(p)
    zs2 = z0 * np.ones_like(p)
    xyz2 = np.c_[xs2, ys2, zs2]

    plot3d(xs1, ys1, zs1, color=(0.0, 0.8, 0.8), tube_radius=0.1)
    plot3d(xs2, ys2, zs2, color=(0.8, 0.8, 0.0), tube_radius=0.1)
    savefig('knot.png')
    show()

    print computeLinkingNumber(xyz1, xyz2)
