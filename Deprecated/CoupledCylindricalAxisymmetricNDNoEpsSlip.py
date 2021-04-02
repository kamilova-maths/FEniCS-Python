from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# Define mesh and geometry
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 60, 60)

a = 1.0
#domain = Polygon([Point(1, 0), Point(1, a), Point(0.5, 1), Point(0, 1), Point(0, 0)])
#mesh = generate_mesh(domain, 50)

# Create mesh
n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q1 = FiniteElement("CG", triangle, 1)
Q2 = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q1, Q2]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p, T) = split(w)
(v, q1, q2) = split(TestFunction(W))
# Define the viscosity and bcs
Gamma = Constant(5.0)
Pe = Constant(27.0)
#Pe = Constant(1.08)
Bi = Constant(11.6)
#Bi = Constant(58.0)
Qc = Constant(0.0)
#St = Constant(0.045)
St = Constant(1.0)
#Qc2 = Constant(0.0)
#Qc2 = Expression("(1/(x1-x2))*(x[1]<x1)*(x[1]>x2)", degree=1,  x1=0.3, x2=0.1)
Qc2 = Expression("Qfun*exp ( -pow( x[1] -( x1-x2 )/2, 2 )/( 2*pow( x1-x2,2 ) ) )", degree=1, Qfun=0.5, x1=0.3, x2=0.1)

Ta = Expression("1-x[1]", degree=1)

mu = exp(-Gamma * T)
u_in = Constant(-4.0) # uin = 4.0 is the value at which the feed in rate is balanced with the electrode consumption.
u_c = Constant(-1.0)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
weird = 'near(x[1], 1.0) && x[0]>=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_slip = DirichletBC(W.sub(0).sub(1), Constant(0.0), weird)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
bcs = [bcu_inflow, bcu_wall, bcu_slip, bcT_inflow, bcu_symmetry]
#Define the variational form
#vsc = as_vector([v[0], asp*v[1]])
#usc = as_vector([u[0], asp*u[1]])
f = Constant((0, -1))

colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero

# We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 5)
CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 1)
CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.1").mark(colors, 2)
# a1 = str(a)
# CompiledSubDomain("near( ( ("+a1+"-1) /0.5)*(x[0] - 1) +" + a1 + "- x[1], 0.0) && x[0]>=0.5").mark(colors, 2)
CompiledSubDomain("near(x[0], 0.2)").mark(colors, 3)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 4)  # outflow

x = SpatialCoordinate(mesh)

# Create the measure
ds = Measure("ds", subdomain_data=colors)


# symmetric gradient
def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))


# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-p*Identity(3)


def cond(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


def sigmabc(v, p):
    return 2*mu*cond(v) - p*Identity(2)


def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)


a1 = (inner(sigma(u, p), epsilon(v))) * x[0] * dx
#a2 = (- (u[0].dx(0) + u[1].dx(0) + (1/x[0]) * u[0]) * q1 - dot(f, v)) * x[0] * dx
a2 = (- div_cyl(u) * q1 - St * dot(f, v)) * x[0] * dx
a3 = (dot(u, grad(T)) * q2 + (1 / Pe) * inner(grad(q2), grad(T)) - Qc2*q2) * x[0] * dx
b1 = - dot(dot(sigmabc(u, p), v), n) * x[0] * ds(1)
b3 = - dot(dot(sigmabc(u, p), v), n) * x[0] * ds(3)
b4 = - dot(dot(sigmabc(u, p), v), n) * x[0] * ds(4)
b5 = - (1 / Pe) * (q2 * Qc * x[0] * ds(4) - Bi * q2 * T * x[0] * ds(3) + q2 * Bi * Ta * x[0] * ds(3)) -\
     dot(grad(T), q2*n)*x[0]*ds(1)
#b5 = - (1 / Pe) * ( - Bi * q2 * T * x[0] * ds(3) + q2 * Bi * Ta * x[0] * ds(3))
F = a1 + a2 + a3 + b5

# Analytic continuation for viscosity - temperature dependence
for Gamma_val in [1, 5, 10, 15, 20, 23]:
    Gamma.assign(Gamma_val)
    print('Gamma =', Gamma_val)
    solve(F == 0, w, bcs)
    # Extract solution
    (u, p, T) = w.split()

    # Extract flux
    # flux = dot(u, n) * dot(u, n) * ds(1)
    # total_flux_old = assemble(flux)

    # print("Total flux on boundary 2 is", total_flux_old)

Qc2 = Expression("Qfun*exp ( -pow( x[1] -( x1-x2 )/2, 2 )/( 2*pow( x1-x2,2 ) ) )", degree=1, Qfun=2.3710, x1=0.3,
                 x2=0.1)
solve(F == 0, w, bcs)

# Plot solutions
File("Results/velocityCylNoEpsSlip.pvd") << u
File("Results/TemperatureCylNoEpsSlip.pvd") << T

W2 = FunctionSpace(mesh, Q2)
Pmu = project(mu, W2)

c = plot(p, title='pressure')
plt.colorbar(c)
plt.show()

c = plot(u, title='velocity')
plt.colorbar(c)
plt.show()

c = plot(T, title='Temperature')
plt.colorbar(c)
plt.show()

c = plot(mu, title='Viscosity, uin = 2.5')
plt.colorbar(c)
plt.show()
