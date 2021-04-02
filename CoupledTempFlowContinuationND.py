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
    #x[:,1] = 0.5 * (cos(pi * (x[:,1] - 1.) / 2.) + 1.)
    x[:, 0] = (x[:, 0] - 0.5) * 2.
    x[:, 0] = 0.5 * (cos(pi * (x[:, 0] - 1.) / 2.) + 1.)
    #Scale
    x[:,0] = x[:,0]*lx
    x[:,1] = x[:,1]*ly

    return m

#mesh = mesh_ref(1.0,1.0, 60,60)
#mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 60, 60)
mesh = Mesh('Meshes/IsothermalRefinedMesh.xml')
#a = Constant(0.2)
#domain = Polygon([Point(1, 0), Point(1, a), Point(0.5, 1), Point(-0.5, 1), Point(-1, a), Point(-1, 0)])
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
#Bi = Constant(58.0)
Bi = Constant(11.6)
#Qc2 = Expression("(1/(x1-x2))*(x[1]<x1)*(x[1]>x2)", degree=2,  x1=0.3, x2=0.1)
#Qfun=2.3710
Qc2 = Expression("Qfun*exp ( -pow( x[1] -(( x1-x2 )/2 + x2), 2 )/( 2*pow( x1-x2,2 ) ) )", degree=1, Qfun=2.3710, x1=0.3,
                 x2=0.1)

Ta = Expression("1-x[1]", degree=1)

mu = exp(-Gamma * T)

u_in = Constant(-2.0)
u_c = Constant(-1.0)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
bcs = [bcu_inflow, bcu_wall, bcT_inflow, bcu_symmetry, bcu_outflow]
# Define the variational form
epsilon = sym(grad(u))
f = Constant((0, -1))

colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero

# We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 4)
CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 0)
CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.1").mark(colors, 1)
CompiledSubDomain("near(x[0], 0.2)").mark(colors, 2)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 3)  # outflow

# Create the measure
ds = Measure("ds", subdomain_data=colors)

I = Identity(2)
F = (2 * mu * inner(epsilon, grad(v)) - div(u) * q1 - div(v) * p - dot(f, v) + dot(u, grad(T)) * q2 + (
       1 / Pe) * inner(grad(q2), grad(T)) - Qc2*q2) * dx \
  - (1 / Pe) * (-Bi * q2 * T * ds(2) + q2 * Bi * Ta * ds(2)) \
  - dot(dot(epsilon, v), n) * ds(0) + dot(dot(p * I, v), n) * ds(0) - dot(dot(epsilon, v), n) * ds(2) \
    + dot(dot(p * I, v), n) * ds(2) - dot(dot(epsilon, v), n) * ds(3) + dot(dot(p * I, v), n) * ds(3)

for Gamma_val in [1, 5, 10, 15, 20, 23]:
    Gamma.assign(Gamma_val)
    print('Gamma =', Gamma_val)
    solve(F == 0, w, bcs)

#Qc2 = Expression("Qfun*exp ( -pow( x[1] -(( x1-x2 )/2 + x2), 2 )/( 2*pow( x1-x2,2 ) ) )", degree=1, Qfun=2.3710, x1=0.3,
 #                x2=0.1)
(u, p, T) = w.split()

flux = dot(u, n) * dot(u, n) * ds(1)
total_flux_new = assemble(flux)
print("Total flux on ds1 is", total_flux_new)

# Plot solutions
File("Results/velocityCartesian.pvd") << u
File("Results/pressureCartesian.pvd") << p
File("Results/TemperatureCartesian.pvd") << T

W2 = FunctionSpace(mesh, Q2)
Pmu = project(mu, W2)

pQc2 = project(Qc2, W2)
c = plot(pQc2)
plt.colorbar(c)
plt.show()

c = plot(p, title='pressure')
plt.colorbar(c)
plt.show()

c = plot(u, title='velocity')
plt.colorbar(c)
plt.show()

c = plot(T, title='Temperature')
plt.colorbar(c)
plt.show()

