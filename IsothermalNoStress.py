from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
# Define mesh and geometry - We solve for half of the domain we need, and impose symmetry

mesh = RectangleMesh(Point(0, 0), Point(1, 1), 100, 100)
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
# mu = Expression('exp(-a*pow(x[0],2))', degree=2, a=10)

u_in = Constant(-2.0)
u_c = Constant(-1.0)


# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<=0.5'
wall = 'near(x[0], 1.0)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]
# Define stress tensor
# epsilon = sym(grad(u))
x = SpatialCoordinate(mesh)


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
f = Constant((0, -1))
# The vectors defined in Fenics are automatically dimensional. We introduce the aspect ratio here, turning the dimensional
# v and u into the non-dimensional ones vsc and usc, noting that usc[0] = (1/asp)*u[0] (or equivalently, we multiply the
# second component by asp.
colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
# We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 5)
CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.5").mark(colors, 1)
CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.5").mark(colors, 2)
CompiledSubDomain("near(x[0], 1.0)").mark(colors, 3)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 4)  # outflow

# Create the measure
ds = Measure("ds", subdomain_data=colors)

a1 = (inner(sigma(u, p), epsilon(v))) * dx
a2 = (- div(u) * q - dot(f, v)) * dx

# For the boundary terms, note that ds(3) is the only one here that varies along x[1], which is where the asp rescaling
# is. Therefore we have to multiply only that term by (1/asp), or equivalently, multiply the other terms by asp.
b1 = - dot(dot(sigmabc(u, p), v), n) * ds(1)
b3 = - dot(dot(sigmabc(u, p), v), n) * ds(3)
b4 = - dot(dot(sigmabc(u, p), v), n) * ds(4)
F = a1 + a2 + b1 + b3 + b4

# Solve problem
solve(F == 0, w, bcs)

# Plot solutions
(u, p) = w.split()
File("Results/velocityIsothermalNoStress.pvd") << u

c = plot(u, title='velocity')
plt.colorbar(c)
plt.show()