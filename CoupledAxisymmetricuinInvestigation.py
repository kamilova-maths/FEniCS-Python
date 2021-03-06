"""CoupledAxisymmetricuinInvestigation

This script performs a small parameter sweep for the inlet velocity. This is based on the CoupledAxisymmetric code.

The results are saved and plotted externally.

This script requires that fenicsproject be installed within the Python
environment you are running this script in, as well as numpy and matplotlib.

"""

from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np


# Define viscosity function
def mu(T_local=None):
    if T_local==None:
        T_local = T
    return exp(-Gamma*T_local)


# Define symmetric gradient for Cartesian coordinates
def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# Define symmetric gradient for cylindrical coordinates
def epsilon_cyl(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))


# Define stress tensor for Cartesian coordinates
def sigma(v, p):
    return 2*mu()*epsilon(v)-p*Identity(2)


# Define stress tensor for cylindrical coordinates
def sigma_cyl(v, p):
    return 2*mu()*epsilon_cyl(v)-Id(p)


# Define a 3D pressure matrix
def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])


# Define the divergence in cylindrical coordinates
def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)


# Define model non-dimensional parameters
Gamma = Constant(10.0)
Pe = Constant(27.0)
Bi = Constant(11.6)


# Cylindric
u_in_cyl = Constant(-4.0)

# Casing velocity
u_c = Constant(-1.0)

# Define the forcing term (can also be zero)
f = Constant((0, -1))

# Specify a mesh
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 60, 60)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
S = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q, S]))

# Define the boundaries
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'

# number of uin values to test
iterations = 6

# initialise the stress values arrays
stress_values_0 = []
stress_values_1 = []
stress_values_3 = []

# normal vector
n = FacetNormal(mesh)

# Define Function and TestFunction(s)
w = Function(W)
(u, p, T) = split(w)
(v, q, s1) = split(TestFunction(W))
x = SpatialCoordinate(mesh)

# Define the heating terms, namely the induction function and the wall heating
Qc2 = Expression("Qfun*exp ( -pow( x[1] -(( x1-x2 )/2 + x2), 2 )/( 2*pow( x1-x2,2 ) ) )", degree=3, Qfun=2.3710, x1=0.3,
                x2=0.1)

Ta = Expression("1-x[1]", degree=1)

# Specify the inlet velocity values that will be used
uin_values = [-2.0, -2.5, -3.0, -3.5, -4.0, -4.5]

# Define the boundary conditions
bcu_inflow_cyl = DirichletBC(W.sub(0), (0.0, u_in_cyl), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcP_outflow = DirichletBC(W.sub(1), 0.0, outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)

# Define facets
facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)  # FACET function

# Define the subdomains for the integration measure
CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)

# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)


# Cylindrical - variational problem
a_cyl = (inner(sigma_cyl(u, p), epsilon_cyl(v)) - div_cyl(u) * q + (dot(u, grad(T)) * s1 + (
        1 / Pe) * inner(grad(s1), grad(T)))) * x[0] * dx() - (1 / Pe) * (-Bi * s1 * T * x[0] * ds(2))
b0 = - dot(dot(sigma(u, p), v), n) * x[0] * ds(0) - s1 * dot(grad(T), n) * x[0] * ds(0)
b2 = - dot(dot(sigma(u, p), v), n) * ds(2)
b3 = -dot(dot(sigma(u, p), v), n) * x[0] * ds(3)
b4 = + dot(p * n, v) * ds(4)

L_cyl = (- dot(f, v) - Qc2 * s1) * x[0] * dx() - (1 / Pe) * (s1 * Bi * Ta * x[0] * ds(2))

F_cyl = a_cyl + L_cyl + b0 + b2 + b3 + b4

dw = TrialFunction(W)

# Jacobian
J_cyl = derivative(F_cyl, w, dw)

# Beginning the iterative procedure for the parameter sweep of uin_values
for i in range(len(uin_values)):
    u_in = uin_values[i]
    print('u_in = ', u_in)


    bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
    # Impose boundary conditions

    bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry, bcT_inflow]
    bcs_cyl = [bcu_inflow_cyl, bcu_wall, bcu_outflow, bcu_symmetry, bcT_inflow]

    # Analytic continuation required to reach Gamma = 23 in the stiff non-linear viscosity function
    for Gamma_val in [1, 10, 15, 20, 23]:
        Gamma.assign(Gamma_val)
        print('Gamma =', Gamma_val)

        # Define and save problem
        problem = NonlinearVariationalProblem(F_cyl, w, bcs_cyl, J_cyl)

        solver = NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["linear_solver"] = 'umfpack'

        solver.solve()
        # Extract solution
        (u, p, T) = w.split()

   # Computing the stress tensor for post-processing calculations
    Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
    sig = Function(Vsig, name="Stress")
    sig.assign(project(sigma(u, p), Vsig))
    File("Results/StressTensorCartesianCDuin" + str(u_in) + ".pvd") << sig

    # Computing the integrals of the normal stress at each boundary
    area0 = assemble(1.0 * ds(0))
    normal_stress0 = assemble(inner(sig * n, n) * ds(0)) / area0
    stress_values_0.append(normal_stress0)

    area1 = assemble(1.0 * ds(1))
    normal_stress1 = assemble(inner(sig * n, n) * ds(1)) / area1
    stress_values_1.append(normal_stress1)

    area3 = assemble(1.0 * ds(3))
    normal_stress3 = assemble(inner(sig * n, n) * ds(3)) / area3
    stress_values_3.append(normal_stress3)

# Saving the obtained stress integral values to a file
values = np.asarray([uin_values, stress_values_0, stress_values_1, stress_values_3])
np.savetxt("Results/NormalStressTopCartesianChanginguin.csv", values.T, delimiter='\t')