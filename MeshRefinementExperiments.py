from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
from numpy import cos , pi

# Define mesh and geometry
def mesh_ref(lx,ly, Nx,Ny):
    m = UnitSquareMesh(Nx, Ny)
    x = m.coordinates()

    #Refine on top and bottom sides
    #x[:,1] = (x[:,1] - 0.5) * 2.
    #x[:,1] = 0.5 * (cos(pi * (x[:,1] - 1.) / 2.) + 1.)
    x[:, 0] = (x[:, 0] - 0.5) * 2.
    x[:, 0] = 0.5 * (cos(pi * (x[:, 0] - 1.) / 2.) + 1.)
    #Scale
    x[:,0] = x[:,0]*lx
    x[:,1] = x[:,1]*ly

    return m


#mesh = mesh_ref(1.0,1.0, 60,60)
#mesh = RectangleMesh(Point(0, 0), Point(1, 1), 60, 60)
a = Constant(0.7)
b = Constant(0.8)
domain = Polygon([Point(1, 0), Point(1, a), Point(b, (a+1)/2), Point(0.5, 1), Point(0, 1), Point(0, 0)])
mesh = generate_mesh(domain, 50)

plot(mesh)
plt.show()
