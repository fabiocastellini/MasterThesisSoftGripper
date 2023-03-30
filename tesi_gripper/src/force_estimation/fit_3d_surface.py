import matplotlib
from matplotlib import cm  # to colormap 3D surfaces from blue to red
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pyvista as pv
import math

graphWidth = 800 # units are pixels
graphHeight = 600 # units are pixels

# 3D contour plot lines
numberOfContourLines = 16


def SurfacePlot(func, data, fittedParameters):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)

    matplotlib.pyplot.grid(True)
    axes = Axes3D(f)

    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    xModel = numpy.linspace(min(x_data), max(x_data), 20)
    yModel = numpy.linspace(min(y_data), max(y_data), 20)
    X, Y = numpy.meshgrid(xModel, yModel)

    Z = func(numpy.array([X, Y]), *fittedParameters)

    axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)

    axes.scatter(x_data, y_data, z_data) # show data along with plotted surface

    axes.set_title('Surface Plot (click-drag with mouse)') # add a title for surface plot
    axes.set_xlabel('X Data') # X axis data label
    axes.set_ylabel('Y Data') # Y axis data label
    axes.set_zlabel('Z Data') # Z axis data label

    plt.show()
    plt.close('all') # clean up after using pyplot or else there can be memory and process problems


def ScatterPlot(data):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)

    matplotlib.pyplot.grid(True)
    axes = Axes3D(f)
    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    axes.scatter(x_data, y_data, z_data)

    axes.set_title('Scatter Plot (click-drag with mouse)')
    axes.set_xlabel('X Data')
    axes.set_ylabel('Y Data')
    axes.set_zlabel('Z Data')

    plt.show()
    plt.close('all') # clean up after using pyplot or else there can be memory and process problems


def paraBolEqn(data, a, b, c, d):
    x = data[0]
    y = data[1]
    return -(((x - b) / a) ** 2 + ((y - d) / c) ** 2) + 1.0

def sphereEqn(data, a, b, c, r):
    x = data[0]
    y = data[1]
    return np.sqrt(abs( (x-a)**2 + (y-b)**2 - r**2 )) + c


def sphereFit(spX, spY, spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX), 4))
    A[:, 0] = spX * 2
    A[:, 1] = spY * 2
    A[:, 2] = spZ * 2
    A[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX), 1))
    f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ)
    C, residules, rank, singval = np.linalg.lstsq(A, f)

    #   solve for the radius
    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]


def create_mesh_from_points(points):

    points = np.array(points)
    initial_points = points[:,0,:]
    final_points = points[:,-1,:]

    """
    augmented_points = []
    num_points_to_add = 5  # interpolation points to add in between real points
    for j in range(len(final_points)-1):
        p = final_points[j]
        q = final_points[j + 1]

        print("-------------")
        print(p)
        print(q)

        augmented_points.append(p)

        x_increment = (q[0]-p[0])/(num_points_to_add+1)
        y_increment = (q[1]-p[1])/(num_points_to_add+1)
        z_increment = (q[2]-p[2])/(num_points_to_add+1)

        for k in range(num_points_to_add):
            interpolated_point = [ p[0]+x_increment*(k+1), p[1]+y_increment*(k+1), p[2]+z_increment*(k+1) ]
            augmented_points.append(interpolated_point)
            print(interpolated_point)
        print("--------------")
    augmented_points = np.array(augmented_points)
    """


    # points is a 3D numpy array (n_points, 3) coordinates of a sphere
    cloud = pv.PolyData(initial_points)
    initial_volume = cloud.delaunay_3d()

    cloud = pv.PolyData(final_points)
    final_volume = cloud.delaunay_3d()

    vol_1 = initial_volume.extract_geometry()
    vol_2 = final_volume.extract_geometry()

    p = pv.Plotter(shape=(1, 2))
    p.add_mesh(vol_1, color='orange', opacity=1, show_edges=True)
    p.subplot(0, 1)
    p.add_mesh(vol_2, color='cyan', opacity=1, show_edges=True)
    p.link_views()
    p.show()

    # --------------
    result = vol_1.boolean_union(vol_2)
    if result.n_points > 0:
        pl = pv.Plotter()
        _ = pl.add_mesh(vol_1, color='orange', style='wireframe', line_width=3)
        _ = pl.add_mesh(vol_2, color='cyan', style='wireframe', line_width=3)
        _ = pl.add_mesh(result, color='tan')

        pl.camera_position = 'xz'
        pl.show()

    # ---------------
    result = vol_1.boolean_difference(vol_2)
    if result.n_points > 0:
        pl = pv.Plotter()
        _ = pl.add_mesh(vol_1, color='orange', style='wireframe', line_width=3)
        _ = pl.add_mesh(vol_2, color='cyan', style='wireframe', line_width=3)
        _ = pl.add_mesh(result, color='tan')
        pl.camera_position = 'xz'
        pl.show()



def fit_3d_surface_main(points, xf_cm, yf_cm, zf_cm):

    rcParams['font.family'] = 'serif'

    points = np.array(points)

    #data = points[:, 0, :]  #get only initial configuration
    data = points[:, -1, :]   #get only final configuration
    xData, yData, zData = data[:, 0].T, data[:, 1].T, data[:, 2].T

    r, x0, y0, z0 = sphereFit(xData, yData, zData)
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v) * r
    y = np.sin(u) * np.sin(v) * r
    z = np.cos(v) * r
    x = x + x0
    y = y + y0
    z = z + z0

    #   3D plot of Sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xData, yData, zData, zdir='z', s=20, c='b', rasterized=True)
    ax.scatter3D(xf_cm, yf_cm, zf_cm, c='cyan')
    ax.plot_wireframe(x, y, z, color="r")
    ax.set_aspect('auto')
    ax.set_xlim3d(-35, 35)
    ax.set_ylim3d(-35, 35)
    ax.set_zlim3d(-70, 0)
    ax.set_xlabel('$x$ (mm)', fontsize=16)
    ax.set_ylabel('\n$y$ (mm)', fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)', fontsize=16)
    plt.show()
    #plt.savefig('steelBallFitted.pdf', format='pdf', dpi=300, bbox_extra_artists=[zlabel], bbox_inches='tight')
