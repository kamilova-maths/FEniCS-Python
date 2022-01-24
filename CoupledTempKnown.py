"""CoupledTempKnown

This script performs a test for the coupled problem where we impose a temperature solution and verify that the
 resulting velocity is comparable to the asymptotic solutions from Chapter 5 in my thesis.

The results are saved and plotted externally.

This script requires that fenicsproject be installed within the Python
environment you are running this script in, as well as numpy and matplotlib.

"""

from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# Define a mesh
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 80, 80)

# Define the facet normal
n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)

W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Define non-dimensional parameters of the problem
Gamma = Constant(23.0)
Pe = Constant(27.0)
Bi = Constant(11.6)

# Cartesian - We must modify the inlet and outlet velocities so that they  match the rescaled values in EVV chapter
u_in = Constant(-0.5)

# Cylindric
u_in_cyl = Constant(-4.0)

# Casing velocity
u_c = Constant(-1.0)

# Define boundaries of the problem
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'

# Define boundary conditions
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_inflow_cyl = DirichletBC(W.sub(0), (0.0, u_in_cyl), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcP_outflow = DirichletBC(W.sub(1), 0.0, outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]
bcs_cyl = [bcu_inflow_cyl, bcu_wall, bcu_outflow, bcu_symmetry]
x = SpatialCoordinate(mesh)

# Define the temperature expression to match the exact solution used in the Extreme Viscosity Chapter. Must be rescaled
# with the aspect ratio, as per the chapter details

T = Expression("x[0]*x[0]/(asp*asp)", asp=0.2, degree=2)

Texp = "Quaduinp5" # Indicator for the file name to show which experiment is being conducted


# Define viscosity function
def mu(T_local=None):
    if T_local==None:
        T_local = T
    return exp(-Gamma*T_local)


# Define the symmetric gradient
def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1), 0],
                          [v[1].dx(0), v[1].dx(1), 0],
                          [0, 0, 0]]))



# Define the symmetric cylindrical gradient
def epsilon_cyl(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))


# Define the stress tensor
def sigma(v, p):
    return 2*mu()*epsilon(v)-Id(p)


# Define the stress tensor in cylindrical coordinates
def sigma_cyl(v, p):
    return 2*mu()*epsilon_cyl(v)-Id(p)


# Define the pressure matrix
def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])


# Define the symmetric gradient applied at a boundary
def epsilon2d(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# Define the stress tensor applied at a boundary
def sigma2d(v, p):
    return 2*mu()*epsilon2d(v) - p*Identity(2)


# Define the divergence in cylindrical coordinates
def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)


# Define the forcing term
f = Constant((0, -1))

# Define facet functions
facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function

# Define the subdomains for the integration measure
CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)

# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)

# Variational problem in Cartesian coordinates

a = (inner(sigma(u, p), epsilon(v)) - div(u) * q) * dx()
b0 = - dot(dot(sigma2d(u, p), v), n) * ds(0)
b2 = - dot(dot(sigma2d(u, p), v), n) * ds(2)
b3 = -dot(dot(sigma2d(u, p), v), n) * ds(3)
b4 = + dot(p*n, v) * ds(4)
b = b0 + b2 + b3 + b4
L = (- dot(f, v)) * dx()
F = a  + b

# Variational problem in cylindrical coordinates
a_cyl = (inner(sigma_cyl(u, p), epsilon_cyl(v)) - div_cyl(u) * q ) * x[0] * dx()
L_cyl = (- dot(f, v) ) * x[0] * dx()
b0_cyl = - dot(dot(sigma2d(u, p), v), n) * x[0] * ds(0)
b2_cyl = - dot(dot(sigma2d(u, p), v), n) * x[0] * ds(2)
b3_cyl = -dot(dot(sigma2d(u, p), v), n) * x[0] * ds(3)
b4_cyl = + dot(p*n, v) * x[0] * ds(4)
b_cyl = b0_cyl + b2_cyl + b3_cyl + b4_cyl

F_cyl = a_cyl + b_cyl

dw = TrialFunction(W)

# Jacobian
J = derivative(F, w, dw)
# Jacobian in cylindrical coordinates
J_cyl = derivative(F_cyl, w, dw)

# Define the files to save the results
vtkfile_u = File("Results/velocityTemp" + Texp + ".pvd")
vtkfile_p = File("Results/pressureTemp" + Texp + ".pvd")
vtkfile_mu = File("Results/ViscosityTemp"+Texp+".pvd")

# Analytic continuation for reaching Gamma=23
for Gamma_val in [10, 15, 20, 23]:
    Gamma.assign(Gamma_val)
    print('Gamma =', Gamma_val)
    # Choose either Cylindrical or Cartesian coordinates. Uncomment the relevant one.
    problem = NonlinearVariationalProblem(F, w, bcs, J)
    #problem = NonlinearVariationalProblem(F_cyl, w, bcs_cyl, J_cyl)

    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["linear_solver"] = 'umfpack'

    solver.solve()

    # Extract solution
    (u, p) = w.split()
    vtkfile_u << (u, Gamma_val)
    vtkfile_p << (p, Gamma_val)
    W2 = FunctionSpace(mesh, Q)
    Pmu = project(mu(), W2)
    vtkfile_mu << (Pmu, Gamma_val)
