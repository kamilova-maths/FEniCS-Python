from dolfin import *
from mshr import *

# Define all the parameters in our model - according to Transfer thesis
# THE DIMENSIONAL PROBLEM DOES NOT WORK
Tin = Constant(343)
rho = Constant(1800)
k = Constant(3)
c = Constant(900)
gr = Constant(10)
u_c = Constant(0.00001)
u_in = Constant(0.00002)
L = Constant(3)
R0 = Constant(0.5)
R1 = Constant(1.0)
mu0 = Constant(10000000000)
h = Constant(7)
Q = Constant(-300)
#gamma = Constant(0.06821)
gamma = Constant(0.01)

# Define mesh and geometry
mesh = RectangleMesh(Point(-R1, 0), Point(R1, L), 120, 60)

# Create mesh
# mesh = generate_mesh(geometry, 32)
n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q1 = FiniteElement("CG", triangle, 1)
Q2 = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q1, Q2]))

# Define Function and TestFunction(s)
w = Function(W);
(u, p, T) = split(w)
(v, q1, q2) = split(TestFunction(W))

Ta = Expression("((Tbot - Ttop)/L)*(L-x[1])+Tbot", Tbot=Constant(393.0), Ttop=Constant(353.0), L=Constant(3.0),
                degree=1)

# Define the viscosity and bcs
mu = mu0 * exp(-gamma * T)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<0.5 && x[0]>-0.5'
wall1 = 'near(x[0], 1.0)'
wall2 = 'near(x[0], -1.0)'

outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall1 = DirichletBC(W.sub(0), (0.0, u_c), wall1)
bcu_wall2 = DirichletBC(W.sub(0), (0.0, u_c), wall2)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcT_inflow = DirichletBC(W.sub(2), Tin, inflow)
bcs = [bcu_inflow, bcu_wall1, bcu_wall2, bcu_outflow, bcT_inflow]

# Define the variational form
epsilon = sym(grad(u))
f = Constant((0, -rho * gr))
# p0 = Expression("1", degree=1)

colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero

CompiledSubDomain("x[0] == 0").mark(colors, 1)
CompiledSubDomain("near(x[1], 1.0) && x[0]>0.5 && x[0]<-0.5").mark(colors, 2)
CompiledSubDomain("x[0] == 1").mark(colors, 3)  # wall 1
CompiledSubDomain("x[0] == -1").mark(colors, 4)  # wall 2
CompiledSubDomain("x[1] == 0").mark(colors, 5)  # outflow

# Create the measure
ds = Measure("ds", subdomain_data=colors)

I = Identity(2)
F = (2 * mu * inner(epsilon, grad(v)) - div(u) * q1 - div(v) * p - dot(f, v) + rho * c * dot(u, grad(T)) * q2 +
     k * inner(grad(q2), grad(T))) * dx - (q2 * Q * ds(5) - h * q2 * T * ds(3) + q2 * h * Ta * ds(3) -
                                           h * q2 * T * ds(4) + q2 * h * Ta * ds(4)) - dot(dot(epsilon, v), n) * ds(
    2) + dot(dot(p * I, v), n) * ds(2) - dot(dot(epsilon, v), n) * ds(3) + dot(dot(p * I, v), n) * ds(3) -\
    dot(dot(epsilon, v), n) * ds(4) + dot(dot(p * I, v), n) * ds(4) - dot(dot(epsilon, v), n) * ds(5) \
    + dot(dot(p * I, v), n) * ds(5)

for gamma_val in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.041, 0.042, 0.0425, 0.043, 0.045, 0.046, 0.047, 0.048,
                  0.049, 0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063,
                  0.064, 0.065, 0.066, 0.067, 0.068, 0.06821]:
    gamma.assign(gamma_val)
    solve(F == 0, w, bcs)

# Plot solutions
(u, p, T) = w.split()
File("Results/velocityContinuationDim.pvd") << u
File("Results/pressureContinuationDim.pvd") << p
File("Results/TemperatureContinuationDim.pvd") << T

W2 = FunctionSpace(mesh, Q2)
Pmu = project(mu, W2)

File("Results/ViscosityContinuationDim.pvd") << Pmu
