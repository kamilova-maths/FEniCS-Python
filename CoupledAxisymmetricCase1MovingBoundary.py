"""CoupledAxisymmetricCase1MovingBoundary

This script tackles our circumvention for the free surface problem, where rather than imposing
the kinetic and dynamic boundary conditions, we impose the kinetic one and calculate how far away we are
from the dynamic one. We impose a flat free surface having moved either up or down, and then we calculate
what the equivalent cylinder addition should be so that this would be a feasible/realistic result.
We use cylindrical coordinates

The results are saved and plotted externally, and correspond to the last sections of Chapter 4 in my thesis.

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


# Define the symmetric cylindrical gradient
def epsilon_cyl(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))


# Define the stress tensor
def sigma_cyl(v, p):
    return 2*mu()*epsilon_cyl(v)-Id(p)


# Identity matrix for pressure
def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])


# Define symmetric gradient for boundary condition
def epsilon2d(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# Define stress tensor for boundary condition
def sigma2d(v, p):
    return 2*mu()*epsilon2d(v) - p*Identity(2)


# Define divergence in cylindrical coordinates
def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)

# Parameters of the problem
Gamma = Constant(10.0)
Pe = Constant(27.0)
Bi = Constant(1.0)

# Inlet velocity
u_in_cyl = Constant(-4.0)

val =  "Bi1" # indicator for saving files. Refers to the specific numerical experiment that I am running

# casing velocity
u_c = Constant(-1.0)

mesh = Mesh('Meshes/CoupledRefinedMeshGamma5Cartesian.xml')

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
S = FiniteElement("CG", triangle, 1)

# Define the boundaries
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'

# Define free surface displacements
Xvalues = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2]

# Parameters that determine the location of the heating temrs
x_1 = 0.3
x_2 = 0.1
x_min = 0
x_max = 1
tol = 1e-10
x = SpatialCoordinate(mesh)

# Imposed induction function
Qc2 = Expression("Qfun*exp ( -pow( x[1] -(( x1-x2 )/2 + x2), 2 )/( 2*pow( x1-x2,2 ) ) )", degree=3, Qfun=2.3710,
                 x1=0.3,
                 x2=0.1)

# initialise vectors for saving relevant data

vel_flux_values = []
normal_stress_values_ds0 = []
normal_stress_values_ds1 = []
normal_stress_values_ds3 = []
coord = "Cyl" # save coordinate system being used

# Starting the shifting of the free surface, according to X_values
for i in range(len(Xvalues)):
    print('X value = ', Xvalues[i])
    n = FacetNormal(mesh)

    W = FunctionSpace(mesh, MixedElement([V, Q, S]))

    # Define Function and TestFunction(s)
    w = Function(W)
    (u, p, T) = split(w)
    (v, q, s1) = split(TestFunction(W))


    # Add the shit to the heating terms
    Qc2.x1 = x_1 + Xvalues[i]
    Qc2.x2 = x_2 + Xvalues[i]
    x_min = 0 + Xvalues[i]
    x_max = 1 + Xvalues[i]
    # Wall heating function
    Ta = conditional(ge(x[1], x_max), tol,  conditional(ge(x[1], x_min), 1-x[1] + Xvalues[i], 1.0-tol))

    # Impose boundary conditions
    bcu_inflow_cyl = DirichletBC(W.sub(0), (0.0, u_in_cyl), inflow)
    bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
    bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
    bcP_outflow = DirichletBC(W.sub(1), 0.0, outflow)
    bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
    bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
    bcs_cyl = [bcu_inflow_cyl, bcu_wall, bcu_outflow, bcu_symmetry, bcT_inflow]

    # Define facets
    facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function

    # create subdomain measures
    CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
    CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
    CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
    CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
    CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)

    # Create the measure
    ds = Measure("ds", domain=mesh, subdomain_data=facet_f)

    # Define variational problem
    a_cyl = (inner(sigma_cyl(u, p), epsilon_cyl(v)) - div_cyl(u) * q + (dot(u, grad(T)) * s1 + (
           1 / Pe) * inner(grad(s1), grad(T)))) * x[0] * dx() - (1 / Pe) * (-Bi * s1 * T * x[0] * ds(2))
    L_cyl = (- Qc2*s1) * x[0] * dx() - (1/Pe) * (s1 * Bi * Ta * x[0] * ds(2))
    b0_cyl = - dot(dot(sigma2d(u, p), v), n) * x[0] * ds(0) - s1*dot(grad(T), n) * x[0] * ds(0)
    b2_cyl = - dot(dot(sigma2d(u, p), v), n) * x[0] * ds(2)
    b3_cyl = -dot(dot(sigma2d(u, p), v), n) * x[0] * ds(3)
    b4_cyl = + dot(p*n, v) * x[0] * ds(4) # imposing symmetry condition

    F_cyl = a_cyl + L_cyl + b0_cyl + b2_cyl + b3_cyl + b4_cyl

    dw = TrialFunction(W)
    # Jacobian
    J_cyl = derivative(F_cyl, w, dw)

    # Analytic continuation for reaching large values of Gamma.
    for Gamma_val in [1, 10, 15, 20, 23]:
        Gamma.assign(Gamma_val)
        print('Gamma =', Gamma_val)
        # Solve problem
        solve(F_cyl == 0, w, bcs_cyl)

        # Extract solution
        (u, p, T) = w.split()

# Save files
    File("Results/velocityCoupledCase1" + coord + "_X" + str(Xvalues[i]) + val + ".pvd") << u
    File("Results/TemperatureCoupledCase1" + coord + "_X" + str(Xvalues[i]) + val + ".pvd") << T
    File("Results/pressureCoupledCase1" + coord + "_X" + str(Xvalues[i]) + val + ".pvd") << p
    Vsig = TensorFunctionSpace(mesh, "DG", degree=1)
    sig = Function(Vsig, name="Stress" + str(Xvalues[i]))
    sig.assign(project(sigma2d(u, p), Vsig))
    File("Results/StressCase1" + coord + "_X" + str(Xvalues[i]) + val + ".pvd") << sig

    W2 = FunctionSpace(mesh, S)
    # Project onto a new space to evaluate the viscosity, as it is calculated with the solution
    Pmu = project(mu(), W2)

    File("Results/ViscosityCoupled" + coord +"_X" + str(Xvalues[i]) + val + ".pvd") << Pmu

    # Calculate fluxes for each displacement
    flux_cyl = dot(u, n) * dot(u, n) * x[0] * ds(1)
    vel_flux_values.append(assemble(flux_cyl))

    # Calculate normal stress at each boundary  (for verification)
    area1 = assemble(1.0 * x[0] * ds(1))
    normal_stress1 = assemble(2*np.pi*inner(sig * n, n) * x[0] * ds(1))
    area0 = assemble(1.0 * x[0] * ds(0))
    area3 = assemble(1.0 * x[0] * ds(3))
    normal_stress0 = assemble(2*np.pi*inner(sig * n, n) * x[0] * ds(0))
    normal_stress3 = assemble(2*np.pi*inner(sig * n, n) * x[0] * ds(3))

    normal_stress_values_ds0.append(normal_stress0)
    normal_stress_values_ds1.append(normal_stress1)
    normal_stress_values_ds3.append(normal_stress3)


# Plot displacement vs the velocity normal flux.
fig = plt.figure()
plt.plot(Xvalues, vel_flux_values)
plt.ylabel('Velocity flux')
plt.xlabel('X values')
plt.title('Sweep over X values, Cylindric')
plt.show()

# Save velocity normal fluxes and displacement
values = np.asarray([Xvalues, vel_flux_values])
np.savetxt("Results/XvsFlux" + coord + val + ".csv", values.T, delimiter='\t')

# Save stress values
values = np.asarray([Xvalues, normal_stress_values_ds0, normal_stress_values_ds1, normal_stress_values_ds3])
np.savetxt("Results/XvsAverageNormalStressTop" + coord + val + ".csv", values.T, delimiter='\t')