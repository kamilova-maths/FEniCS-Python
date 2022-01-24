"""ConvergenceCoupled

This script performs a convergence test for the coupled problem. Since the mesh
changes quadratically at each iteration, it is required to redefine the problem and
boundary conditions at each time.

The results are saved and plotted externally. We expect a quadratic convergence for
the velocity problem, and temperature problem, as well as linear convergence for
the pressure problem. These results hold (with temperature behaving better than estimated)

This script requires that fenicsproject be installed within the Python
environment you are running this script in, as well as numpy and matplotlib.

"""


from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# Define mesh and normal vector
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 2, 2)
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

# Problem parameters
Gamma = Constant(5.0)
Pe = Constant(27.0)
Bi = Constant(11.6)

# Source term and wall heating
Qc2 = Expression("Qfun*exp ( -pow( x[1] -(( x1-x2 )/2 + x2), 2 )/( 2*pow( x1-x2,2 ) ) )", degree=1, Qfun=2.3710, x1=0.3,
                 x2=0.1)
Ta = Expression("1-x[1]", degree=1)

# Inlet velocity and casing velocity (according to mass conservation)
u_in = Constant(-2.0)
u_c = Constant(-1.0)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
# Define the boundaries
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'

# Define the boundary conditions
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
bcs = [bcu_inflow, bcu_wall, bcT_inflow, bcu_symmetry, bcu_outflow]

# forcing term
f = Constant((0, -1))

# Define boundaries used for integration measures
colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 0)
CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.1").mark(colors, 1)
CompiledSubDomain("near(x[0], 0.2)").mark(colors, 2)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 3)  # outflow
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 4)


# viscosity function
def mu(T_local=None):
    if T_local==None:
        T_local = T
    return exp(-Gamma*T_local)


# symmetric gradient
def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))

# stress tensor
def sigma(v, p):
    return 2*mu()*epsilon(v) - p*Identity(2)


# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=colors)

# Create identity matrix for variational form
I = Identity(2)

# Define the variational form
F = (inner(sigma(u, p), epsilon(v)) - div(u) * q1 - div(v) * p - dot(f, v) + dot(u, grad(T)) * q2 + (
       1 / Pe) * inner(grad(q2), grad(T)) - Qc2*q2) * dx \
  - (1 / Pe) * (-Bi * q2 * T * ds(2) + q2 * Bi * Ta * ds(2)) \
  - dot(dot(sigma(u, p), n), v)*ds(0) - dot(dot(sigma(u, p), n), v)*ds(2) - dot(dot(sigma(u, p), n), v)*ds(3)

# Use analytical continuation to solve the first iteration
for Gamma_val in [1, 5, 10, 15, 20, 23]:
    Gamma.assign(Gamma_val)
    print('Gamma =', Gamma_val)
    solve(F == 0, w, bcs)

# Save the results of the first iteration
(u, p, T) = w.split()


N = 6 # number of mesh refinements
h_prev = mesh.hmin() # save previous mesh size
hvalues = [h_prev] # vector of mesh sizes
errors_u = [] # define empty vectors for errors
errors_p = []
errors_T = []

# Define the function space for calculating the stress tensor, an important
# quantity of interest for our convergence results

Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
sig_num = Function(Vsig, name="Stress Numeric")
sig_num.assign(project(sigma(u, p), Vsig))
area1 = assemble(1.0 * ds(1))
normal_stress_average = [assemble(inner(sig_num * n, n) * ds(1)) / area1]

# Name the vtkfiles that will store the  results
vtkfile_u = File('Results/Coupled_meshref_u.pvd')
vtkfile_p = File('Results/Coupled_meshref_p.pvd')
vtkfile_T = File('Results/Coupled_meshref_T.pvd')
vtkfile_stress = File('Results/Coupled_meshref_stress.pvd')


# Mesh refinement loop begins
for i in range(N):
    print(i)
    # All the function spaces, elements, boundary conditions, and measures must be
    # redefined in the same way as before, as the mesh will change size at each iteration

    mesh = refine(mesh)
    n = FacetNormal(mesh)
    hvalues.append(mesh.hmin())
    V2 = FunctionSpace(mesh, V)
    Qp = FunctionSpace(mesh, Q1)
    QT = FunctionSpace(mesh, Q2)
    u_prev = interpolate(u, V2) # interpolating values on mesh i + 1
    p_prev = interpolate(p, Qp)
    T_prev = interpolate(T, QT)
    # Define Function and TestFunction(s)

    # Define Taylor--Hood function space W
    V = VectorElement("CG", triangle, 2)
    Q1 = FiniteElement("CG", triangle, 1)
    Q2 = FiniteElement("CG", triangle, 1)
    W = FunctionSpace(mesh, MixedElement([V, Q1, Q2]))

    w = Function(W)
    (u, p, T) = split(w)
    (v, q1, q2) = split(TestFunction(W))

    # Define the viscosity and bcs
    bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
    bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
    bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
    bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
    bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
    bcs = [bcu_inflow, bcu_wall, bcT_inflow, bcu_symmetry, bcu_outflow]


    colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    colors.set_all(0)  # default to zero

    CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 0)
    CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.1").mark(colors, 1)
    CompiledSubDomain("near(x[0], 0.2)").mark(colors, 2)  # wall
    CompiledSubDomain("near(x[1], 0.0)").mark(colors, 3)  # outflow
    CompiledSubDomain("near(x[0], 0.0)").mark(colors, 4)

    # Create the measure
    ds = Measure("ds", domain=mesh, subdomain_data=colors)

    # Define the variational form
    F = (inner(sigma(u, p), epsilon(v)) - div(u) * q1 - div(v) * p - dot(f, v) + dot(u, grad(T)) * q2 + (
           1 / Pe) * inner(grad(q2), grad(T)) - Qc2*q2) * dx \
      - (1 / Pe) * (-Bi * q2 * T * ds(2) + q2 * Bi * Ta * ds(2)) \
      - dot(dot(sigma(u, p), n), v)*ds(0) - dot(dot(sigma(u, p), n), v)*ds(2) - dot(dot(sigma(u, p), n), v)*ds(3)

    # Analytic continuation to reach Gamma=23 value.
    for Gamma_val in [1, 5, 10, 15, 20, 23]:
        Gamma.assign(Gamma_val)
        print('Gamma =', Gamma_val)
        solve(F == 0, w, bcs)

    (u, p, T) = w.split()

    # Interpolate the solution so that the error can be computed with accuracy
    u_next = interpolate(u, V2)
    p_next = interpolate(p, Qp)
    T_next = interpolate(T, QT)

    # Save errors to specified vectors
    errors_u.append(np.sqrt(assemble(inner(u_next-u_prev, u_next-u_prev)*dx)))
    errors_p.append(np.sqrt(assemble(inner(p_next - p_prev, p_next - p_prev) * dx)))
    errors_T.append(np.sqrt(assemble(inner(T_next - T_prev, T_next - T_prev) * dx)))
    # Calculate and save the stress at the free surface
    Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
    sig_num = Function(Vsig, name="Stress Numeric")
    sig_num.assign(project(sigma(u, p), Vsig))
    area1 = assemble(1.0 * ds(1))
    normal_stress_average.append(assemble(inner(sig_num * n, n) * ds(1)) / area1)

# Save all values in one vector, and then into a .csv file

values = np.asarray([hvalues, errors_u, errors_p, errors_T, normal_stress_average])
np.savetxt("Results/ErrorsConvergenceCoupled.csv", values.T, delimiter='\t')
