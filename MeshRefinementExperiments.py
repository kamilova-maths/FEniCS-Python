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
    x[:,1] = 0.5 * (cos(pi * (x[:,1] - 1.) / 2.) + 1.)
    # x[:, 0] = (x[:, 0] - 0.5) * 2.
    # x[:, 0] = 0.5 * (cos(pi * (x[:, 0] - 1.) / 2.) + 1.)
    #Scale
    x[:,0] = x[:,0]*lx
    x[:,1] = x[:,1]*ly

    return m


mesh = mesh_ref(0.2, 1.0, 10, 10)
#mesh = RectangleMesh(Point(0, 0), Point(1, 1), 60, 60)
# a = Constant(0.8)
# domain = Polygon([Point(0.2, 0), Point(0.2, a), Point(0.1, 1), Point(0, 1), Point(0, 0)])
# mesh = generate_mesh(domain, 5)
plot(mesh)
print("Initial mesh has this number of cells", mesh.num_cells())
plt.show()

mesh = refine(mesh)
plot(mesh)
print("Refined mesh has this number of cells", mesh.num_cells())
plt.show()
#
# mesh = refine(mesh)
# print("Refined mesh has this number of cells", mesh.num_cells())
#
# mesh = refine(mesh)
# print("Refined mesh has this number of cells", mesh.num_cells())
#
# mesh = refine(mesh)
# print("Refined mesh has this number of cells", mesh.num_cells())
#
# mesh = refine(mesh)
# print("Refined mesh has this number of cells", mesh.num_cells())