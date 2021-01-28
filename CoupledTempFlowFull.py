from dolfin import *
from mshr import *

# Define mesh and geometry
mesh = RectangleMesh(Point(-1, 0), Point(1, 1), 120, 60)
#mesh = RectangleMesh(Point(0, 0), Point(1, 1), 60, 60)
#a = Constant(0.2)
#domain = Polygon([Point(1, 0), Point(1, a), Point(0.5, 1), Point(-0.5, 1), Point(-1, a), Point(-1, 0)])
#mesh = generate_mesh(domain, 50)

# Create mesh
# mesh = generate_mesh(geometry, 32)
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
Bi = Constant(58.0)
Qc = Constant(-1.0)

Ta = Expression("1-x[1]", degree=1)

mu = exp(-Gamma * T)

u_in = Constant(-2.0)
u_c = Constant(-1.0)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<0.5 && x[0]>-0.5'
wall1 = 'near(x[0], 1.0)'
wall2 = 'near(x[0], -1.0)'
# centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall1 = DirichletBC(W.sub(0), (0.0, u_c), wall1)
bcu_wall2 = DirichletBC(W.sub(0), (0.0, u_c), wall2)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
bcs = [bcu_inflow, bcu_wall1, bcu_wall2, bcu_outflow, bcT_inflow]
# Define the variational form
epsilon = sym(grad(u))
f = Constant((0, -1))
#p0 = Expression("1", degree=1)

colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
# We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 5)
CompiledSubDomain("near(x[1], 1.0) && x[0]<0.5").mark(colors, 1)
CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.5").mark(colors, 2)
CompiledSubDomain("near(x[0], 1.0)").mark(colors, 3)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 4)  # outflow
CompiledSubDomain("near(x[0], -1.0)").mark(colors, 6)  # wall2
CompiledSubDomain("near(x[1], 1.0) && x[0]<=-0.5").mark(colors, 7)

# Create the measure
ds = Measure("ds", subdomain_data=colors)

I = Identity(2)
F = (2 * mu * inner(epsilon, grad(v)) - div(u) * q1 - div(v) * p - dot(f, v) + dot(u, grad(T)) * q2 + (
       1 / Pe) * inner(grad(q2), grad(T))) * dx \
  - (1 / Pe) * (
         q2 * Qc * ds(4) - Bi * q2 * T * ds(3) + q2 * Bi * Ta * ds(3) - Bi * q2 * T * ds(6) + q2 * Bi * Ta * ds(6)) \
  - dot(dot(epsilon, v), n) * ds(1) + dot(dot(p * I, v), n) * ds(1) - dot(dot(epsilon, v), n) * ds(3) + dot(
  dot(p * I, v), n) * ds(3) \
  - dot(dot(epsilon, v), n) * ds(4) + dot(
  dot(p * I, v), n) * ds(4) - dot(dot(epsilon, v), n) * ds(6) + dot(dot(p * I, v), n) * ds(6)

#for Gamma_val in [10, 11 ,12 ,13 ,14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
#for Gamma_val in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
 #   Gamma.assign(Gamma_val)
solve(F == 0, w, bcs)

# Plot solutions
(u, p, T) = w.split()
File("Results/velocityGamma5NonDim.pvd") << u
File("Results/pressureGamma5NonDim.pvd") << p
File("Results/TemperatureGamma5NonDim.pvd") << T


W2 = FunctionSpace(mesh, Q2)
Pmu = project(mu, W2)

File("Results/ViscosityGamma5NonDim.pvd") << Pmu
