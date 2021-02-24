from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# Define mesh and geometry - We solve for half of the domain we need, and impose symmetry

mesh = RectangleMesh(Point(0, 0), Point(1, 2), 100, 100)
n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Define the viscosity and bcs

mu = Constant(1.0)
u_in = Constant(-2.0) # inlet velocity
u_c = Constant(-1.0) # wall velocity, and outlet velocity

inflow = 'near(x[1], 2.0) && x[0]<=0.5' # left half of top boundary
wall = 'near(x[0], 1.0)' # right boundary
centre = 'near(x[0], 0.0)' # left boundary
outflow = 'near(x[1], 0.0)' # bottom boundary
weird = 'near(x[1], 2.0) && x[0]>=0.5' # right half of top boundary
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcu_slip = DirichletBC(W.sub(0).sub(1), Constant(0.0), weird) # slip condition that should be right but is somehow wrong
bcs = [bcu_inflow, bcu_wall, bcu_symmetry, bcu_slip]

x = SpatialCoordinate(mesh)

# Define stress tensor
def epsilon(v):
   return sym(as_tensor([[v[0].dx(0), v[0].dx(1), 0],
                         [v[1].dx(0), v[1].dx(1), 0],
                         [0, 0, 0]]))

# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-Id(p)

def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])

def cond(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))

def sigmabc(v, p):
    return 2*mu*cond(v) - p*Identity(2)


# Define the variational form
f = Constant((0, -1)) # forcing term ( I have also tried with this being zero, still no convergence)

colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 5)
CompiledSubDomain("near(x[1], 2.0) && x[0]<=0.5").mark(colors, 1) # top left
CompiledSubDomain("near(x[1], 2.0) && x[0]>=0.5").mark(colors, 2) # top right
CompiledSubDomain("near(x[0], 1.0)").mark(colors, 3)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 4)  # outflow

# Create the measure
ds = Measure("ds", subdomain_data=colors)

a1 = (inner(sigma(u, p), epsilon(v))) * dx
a2 = (- div(u) * q - dot(f, v)) * dx

b1 = - dot(dot(sigmabc(u, p), v), n) * ds(1)
b2 = -dot(dot(sigmabc(u, p), v), n) * ds(2)
b3 = - dot(dot(sigmabc(u, p), v), n) * ds(3)
b4 = - dot(dot(sigmabc(u, p), v), n) * ds(4)

F = a1 + a2

# Solve problem
solve(F == 0, w, bcs)

# Plot solutions
(u, p) = w.split()

File("Results/velocityIsothermalSlipRect.pvd") << u
c = plot(u, title='Velocity Isothermal')
plt.colorbar(c)
plt.show()