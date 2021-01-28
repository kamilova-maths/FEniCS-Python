from dolfin import *
import matplotlib.pyplot as plt

# Define mesh and geometry

mesh = RectangleMesh(Point(0, 0), Point(1, 1), 120, 60)

n = FacetNormal(mesh)

# Define Taylor--Hood function space W
Q = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, Q)

# Define Function and TestFunction(s)
T = Function(W)
v2 = TestFunction(W)

# Define the viscosity and bcs

Pe = Constant(27.0)
Bi = Constant(58.0)
Qc = Constant(0.0)

Qc2 = Expression("(1/(x1-x2))*(x[1]<x1)*(x[1]>x2)", degree=1,  x1=0.3, x2=0.1)

Ta = Expression("1-x[1]", degree=1)
# u = Expression(("0", "x[1]*x[1]"),degree=1)
u = Constant((0, -3.0))

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
# inflow = 'near(x[1], 1.0) && x[0]<0.5 && x[0]>-0.5'
inflow = 'near(x[1], 1.0) && x[0]<0.5'
wall = 'near(x[0], 1.0)'
outflow = 'near(x[1], 0.0)'
centre = 'near(x[0], 0.0)'
bcT_inflow = DirichletBC(W, Constant(0.0), inflow)
# bcu_sym = DirichletBC(W.sub(0), 0.0, centre)
# bcT_sym = DirichletBC(W.sub(2), 0.0, centre)
bcs = bcT_inflow

# Define the variational form

colors = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
colors.set_all(0) # default to zero

CompiledSubDomain("x[0] == 0").mark(colors, 1)
CompiledSubDomain("near(x[1], 1.0) && x[0]>0.5").mark(colors, 2)
CompiledSubDomain("x[0] == 1").mark(colors, 3) # wall
CompiledSubDomain("x[1] == 0").mark(colors, 5) # outflow

# Create the measure
x = SpatialCoordinate(mesh)

ds = Measure("ds", subdomain_data=colors)


F = (dot(u, grad(T))*v2 + (1/Pe)*inner(grad(v2), grad(T)) - v2*Qc2)*x[0]*dx - (1/Pe)*(v2*Qc*x[0]*ds(5) - Bi*v2*T*x[0]*ds(3) + v2*Bi*Ta*x[0]*ds(3)) - dot(grad(T), v2*n)*x[0]*ds(1)

# Solve problem
solve(F == 0, T, bcs)

# Plot solutions
File("Results/TemperatureUncoupledCyl.pvd") << T
plot(T)
plt.show()
