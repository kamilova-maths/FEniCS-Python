"""AutoAdaptiveCoupled

This script runs an adaptive mesh algorithm for the coupled velocity,
pressure, and temperature problem in cylindrical coordinates.

The Stokes problem contains an inherent non-removable corner singularity,
and the adaptive mesh refines the nodes near those corners so that the final
results are as smooth as possible.

This script requires that fenicsproject be installed within the Python
environment you are running this script in, as well as numpy and matplotlib.

"""

from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# Choose initial mesh
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 40, 40, "crossed")

n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
S = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q, S]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p, T) = split(w)
(v, q, s1) = split(TestFunction(W))

# Define the induction heating, and boundary conditions
Qc2 = Expression("Qfun*exp ( -pow( x[1] -(( x1-x2 )/2 + x2), 2 )/( 2*pow( x1-x2,2 ) ) )", degree=3, Qfun=2.3710, x1=0.3,
                x2=0.1)
Qc2.Qfun = 2*2.3710
Ta = Expression("1-x[1]", degree=1)

# Define the constants required for definitions
Gamma = Constant(5.0)
Pe = Constant(27.0)
Bi = Constant(11.6)

# For cylindric coordinates
u_in_cyl = Constant(-4.0)

# casing velocity
u_c = Constant(-1.0)


# Define boundaries
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'

# Define boundary conditions for each variable of interest
bcu_inflow_cyl = DirichletBC(W.sub(0), (0.0, u_in_cyl), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
bcs_cyl = [bcu_inflow_cyl, bcu_wall, bcu_outflow, bcu_symmetry, bcT_inflow]

x = SpatialCoordinate(mesh)


# define the nonlinear viscosity function
def mu(T_local=None):
    if T_local==None:
        T_local = T
    return exp(-Gamma*T_local)



# Symmetric gradient for cylindrical coordinates
def epsilon_cyl(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))



# stress tensor for cylindrical coordinates
def sigma_cyl(v, p):
    return 2*mu()*epsilon_cyl(v)-Id(p)


# pressure matrix
def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])


# symmetric gradient for boundary condition
def cond(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# stress tensor for boundary condition
def sigmabc(v, p):
    return 2*mu()*cond(v) - p*Identity(2)


# divergence for cylindrical coordinates
def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)


# source term
f = Constant((0, -1))

# Mesh facets on which boundary conditions will be applied
facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function

CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0) # inlet
CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1) # free surface
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4) # centre

# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)

# Define the variational form in cylindrical coordinates

a_cyl = (inner(sigma_cyl(u, p), epsilon_cyl(v)) - div_cyl(u) * q + (dot(u, grad(T)) * s1 + (
       1 / Pe) * inner(grad(s1), grad(T)))) * x[0] * dx() - (1 / Pe) * (-Bi * s1 * T * x[0] * ds(2))
L_cyl = (- dot(f, v) - Qc2*s1) * x[0] * dx() - (1/Pe) * (s1 * Bi * Ta * x[0] * ds(2))


F_cyl = a_cyl + L_cyl

# Define error tolerance
tol = 1.e-7

dw = TrialFunction(W)

# Jacobian
J_cyl = derivative(F_cyl, w, dw)

# Minimisation function
M_cyl = inner(w[0], w[0]) * x[0] * dx()


problem_cyl = NonlinearVariationalProblem(F_cyl, w, bcs_cyl, J_cyl)
solver = AdaptiveNonlinearVariationalSolver(problem_cyl, M_cyl)
solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "umfpack"
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
solver.solve(tol)

solver.summary()

#Plot solutions from initial mesh and refined mesh
(u0, p0, T0) = w.root_node().split()
(u1, p1, T1) = w.leaf_node().split()

# Plot the results of the adaptive meshing algorithm
R = w.leaf_node().function_space()
plot(R.mesh())
plt.show()


plot(u0, title="Velocity on initial mesh")
plt.show()
plot(u1, title="Velocity on final mesh")
plt.show()

plot(p0, title="Pressure on initial mesh")
plt.show()
plot(p1, title="Pressure on final mesh")
plt.show()

plot(T0, title="Temperature on initial mesh")
plt.show()
plot(T1, title="Temperature on final mesh")
plt.show()