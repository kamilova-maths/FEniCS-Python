from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np


a = 1
u_in = Constant(-4.0)
u_c = Constant(-1.0)


# symmetric gradient
def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))



# stress tensor
def sigma(v, p):
    return 2 * mu * epsilon(v) - p * Identity(3)


def cond(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


def sigmabc(v, p):
    return 2 * mu * cond(v) - p * Identity(2)


def div_cyl(v):
    return (1 / x[0]) * (x[0] * v[0]).dx(0) + v[1].dx(1)


abnd = str(a)
domain = Polygon([Point(0.2, 0), Point(0.2, a), Point(0.1, 1), Point(0, 1), Point(0, 0)])
mesh = generate_mesh(domain, 100)

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
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
#bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_slip = DirichletBC(W.sub(0).sub(1), Constant(0.0), weird)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcs = [bcu_inflow, bcu_wall, bcu_slip, bcu_symmetry]
# Define the variational form

f = Constant((0, -1))

colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
# We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 5)
CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 1)
CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.1").mark(colors, 2)
#CompiledSubDomain("near( ( ("+abnd+"-1) /0.1)*(x[0] - 0.2) +" + abnd + "- x[1], 0.0) && x[0]>=0.1").mark(colors, 2)
CompiledSubDomain("near(x[0], 0.2)").mark(colors, 3)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 4)  # outflow

x = SpatialCoordinate(mesh)

# Create the measure
ds = Measure("ds", subdomain_data=colors)

a1 = (inner(sigma(u, p), epsilon(v))) * x[0] * dx
a2 = (- div_cyl(u) * q1 - dot(f, v)) * x[0] * dx
b1 = - dot(dot(sigmabc(u, p), v), n) * x[0] * ds(1)
b3 = - dot(dot(sigmabc(u, p), v), n) * x[0] * ds(3)
b4 = - dot(dot(sigmabc(u, p), v), n) * x[0] * ds(4)
F = a1 + a2

solve(F == 0, w, bcs)
# Extract solution
(u, p) = w.split()
ones = as_vector([1.0, 1.0])
# OK I think this works.
stress = 2*mu*dot(dot(sigmabc(u, p), ones),n)*x[0] *ds(2)
#stress = 2*mu*(u[1].dx(1)) * x[0] * ds(2)
total_stress_ds2 = assemble(stress)

print("Total stress on boundary 2 is", total_stress_ds2)
