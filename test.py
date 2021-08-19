import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


def intersect(points, dir_vector) -> np.ndarray:
    """
    Calculate the least squares solution of the point closest
    to all lines defined by the gps positions and the angles
    to the lamppost. https://stackoverflow.com/a/52089867

    :return: point of (nearest) intersection
    :rtype: np.ndarray
    """

    # dir_vector = np.asarray([[math.cos(rad[0]) * math.cos(rad[1]),
    #                           math.sin(rad[0]) * math.cos(rad[1]),
    #                           math.sin(rad[1])] for rad in self.rads])

    uv = dir_vector / np.sqrt((dir_vector ** 2).sum(-1))[..., np.newaxis]
    print(uv)

    projs = np.eye(uv.shape[1]) - uv[:, :, np.newaxis] * uv[:, np.newaxis]

    R = projs.sum(axis=0)
    q = (projs @ points[:, :, np.newaxis]).sum(axis=0)

    p = np.linalg.lstsq(R, q, rcond=-1)[0]
    return p


def intersect_ex(P0, P1):
    """P0 and P1 are NxD arrays defining N lines.
    D is the dimension of the space. This function
    returns the least squares intersection of the N
    lines from the system given by eq. 13 in
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
    """
    # generate all line direction vectors
    n = (P1 - P0) / np.linalg.norm(P1 - P0, axis=1)[:, np.newaxis]  # normalized
    print(n)

    # generate the array of all projectors
    projs = np.eye(n.shape[1]) - n[:, :, np.newaxis] * n[:, np.newaxis]  # I - n*n.T
    # see fig. 1

    # generate R matrix and q vector
    R = projs.sum(axis=0)
    q = (projs @ P0[:, :, np.newaxis]).sum(axis=0)

    # solve the least squares problem for the
    # intersection point p: Rp = q
    p = np.linalg.lstsq(R, q, rcond=None)[0]

    return p


# n = 6
# P0 = np.stack((np.array([5,5])+3*np.random.random(size=2) for i in range(n)))
# a = np.linspace(0,2*np.pi,n)+np.random.random(size=n)*np.pi/5.0
# P1 = np.array([5+5*np.sin(a),5+5*np.cos(a)]).T

P0 = np.array([[0, 0, 0], [1, 1, 0]])
P1 = np.array([[1, 1, 1], [0, 0, 1]])

x0, z0, y0 = 0, 0, 0
x1, z1, y1 = 1, 1, 1
x2, z2, y2 = 1, 1, 0
x3, z3, y3 = 0, 0, 1


print(intersect(P0, np.array([[1, 1, 1], [-1, -1, 1]])))
print(intersect_ex(P0, P1))

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
P = intersect(P0, np.array([[1, 1, 1], [-1, -1, 1]]))
P = P.ravel()

ax.plot([x0, x1], [z0, z1], [y0, y1])
ax.plot([x2, x3], [z2, z3], [y2, y3])
ax.scatter(P[0], P[1], P[2])
# ax.plot(x0, z0, y0)

plt.show()
