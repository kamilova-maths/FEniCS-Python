from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

u_in = Constant(-2.0)
u_c = Constant(-1.0)


# symmetric gradient
def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))



# stress tensor
def sigma(v, p):
    return 2 * mu * epsilon(v) - p * Identity(2)


#mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 250, 50, "crossed")
#mesh = Mesh('Meshes/IsothermalTopRefinedMesh.xml')
mesh = Mesh('Meshes/IsothermalAdaptiveMesh.xml')
mesh_type = "Adaptive"
# Create mesh
n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q1 = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q1]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p) = split(w)
(v, q1) = split(TestFunction(W))
# Define the viscosity and bcs

mu = Constant(1.0)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
weird = 'near(x[1], 1.0) && x[0] >=0.1'
#weird = 'near( ( ('+abnd+'-1) /0.1)*(x[0] - 0.2) +' + abnd + '- x[1], 0.0) && x[0]>=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
pinpoint = 'near(x[1], 1.0) && near(x[0], 0.1)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
#bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_slip = DirichletBC(W.sub(0).sub(1), Constant(0.0), weird)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcs = [bcu_inflow, bcu_wall, bcu_slip, bcu_symmetry]
# Define the variational form


colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
# We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 4)
CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 0)
CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.1").mark(colors, 1)
CompiledSubDomain("near(x[0], 0.2)").mark(colors, 2)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 3)  # outflow

x = SpatialCoordinate(mesh)

# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=colors)

a1 = (inner(sigma(u, p), epsilon(v))) * dx
a2 = (- div(u) * q1 ) * dx
b1 = - dot(dot(sigma(u, p), v), n) * ds(1)
b3 = - dot(dot(sigma(u, p), v), n) * ds(3)
b4 = - dot(dot(sigma(u, p), v), n) * ds(4)
F = a1 + a2

solve(F == 0, w, bcs)
# Extract solution
(u, p) = w.split()
Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
sig = Function(Vsig, name="Stress")
sig.assign(project(sigma(u, p), Vsig))

File("Results/StressIsothermal" + mesh_type + "Slip.pvd") << sig

File("Results/velocityIsothermal" + mesh_type + "Slip.pvd") << u
File("Results/pressureIsothermal" + mesh_type + "Slip.pvd") << p

c = plot(p)
plt.colorbar(c)
plt.show()
# x0val_right = np.linspace(0.1, 0.2, 200)
# x0val_left = x0val_right - 0.1
# pds0 = []
# pds1 = []
#
# for i in range(len(x0val_right)):
#     pds0.append(p(x0val_left[i], 1.0))
#     pds1.append(p(x0val_right[i], 1.0))
#
# values = np.asarray([x0val_left, pds0, x0val_right, pds1])
# np.savetxt("Results/IsothermalFreeSlipPTop.csv", values.T, delimiter='\t')
