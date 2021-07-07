from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
# Define mesh and geometry - We solve for half of the domain we need, and impose symmetry
#domain = Polygon([Point(1, 0), Point(1, 0.5), Point(0.5, 1), Point(0, 1), Point(0, 0)])
#mesh = generate_mesh(domain, 100)
#mesh = RectangleMesh(Point(0, 0), Point(0.2, 1.0), 250, 50, "crossed")
#mesh = RectangleMesh(Point(0, 0), Point(0.2, 1.0),80, 80, "crossed")
#mesh = Mesh('Meshes/Experiment.xml')
mesh = Mesh('Meshes/IsothermalAdaptiveMesh.xml')

#mesh = Mesh('Meshes/IsothermalPressureRefinedMesh.xml')
mesh_ind = "Original"
n = FacetNormal(mesh)

plot(mesh)
plt.show()
# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2) # original spaces
Q = FiniteElement("CG", triangle, 1) # original spaces
W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Define the viscosity and bcs

mu = Constant(1.0)


u_in = Constant(-4.0)
u_c = Constant(-1.0)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<=0.1' # This is C++ code.
right = 'near(x[1], 1.0) && x[0]>=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
pinpoint = 'near(x[1], 0.0) && near(x[0], 0.2)'
#bcp_pinpoint = DirichletBC(W.sub(1), Constant(1.0), pinpoint, "pointwise")
#bcp_right = DirichletBC(W.sub(1), Constant(0.0), right)
#bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcp_inflow = DirichletBC(W.sub(1), 50.0, inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcP_right = DirichletBC(W.sub(1), 0.0, right)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcs = [bcp_inflow, bcu_wall, bcu_outflow, bcu_symmetry, bcP_right]

x = SpatialCoordinate(mesh)


def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-p*Identity(2)


# symmetric cylindric gradient
def epsilon_cyl(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))

# stress tensor
def sigma_cyl(v, p):
    return 2*mu*epsilon_cyl(v)-Id(p)


def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])


def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)



# Define the variational form

facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function
#colors.set_all(0)  # default to zero
#We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)
# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
a1 = (inner(sigma_cyl(u, p), epsilon_cyl(v))) * x[0] * dx
a2 = (- div_cyl(u) * q ) * x[0] * dx
b1 = - dot(dot(sigma(u, p), n), v)* x[0] * ds(0) - dot(dot(sigma(u, p), n), v)* x[0] * ds(2) - dot(dot(sigma(u, p), n), v)* x[0] *ds(3)

F = a1 + a2 + b1
# Solve problem
solve(F == 0, w, bcs)

# Plot solutions
(u, p) = w.split()

Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
sig = Function(Vsig, name="Stress")
sig.assign(project(sigma(u, p), Vsig))

normal_stress_0 = assemble(2*np.pi*inner(sig*n, n)*x[0]*ds(0))
print("Normal stress with mesh " + mesh_ind + " is ", normal_stress_0 )

flux = dot(u, n) * dot(u, n) * ds(1)
print("flux is ", assemble(flux))
sav = 1.0
if sav == 1.0:
    File("Results/IsothermalCyl" + mesh_ind + "NormalStress.pvd") << sig
    File("Results/velocityCyl" + mesh_ind + "IsothermalNoStress.pvd") << u
    File("Results/pressureCyl" + mesh_ind + "IsothermalNoStress.pvd") << p
