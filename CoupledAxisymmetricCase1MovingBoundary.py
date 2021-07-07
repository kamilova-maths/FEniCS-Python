from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

def mu(T_local=None):
    if T_local==None:
        T_local = T
    return exp(-Gamma*T_local)


def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1), 0],
                          [v[1].dx(0), v[1].dx(1), 0],
                          [0, 0, 0]]))


# symmetric cylindric gradient
def epsilon_cyl(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))


# stress tensor
def sigma(v, p):
    return 2*mu()*epsilon(v)-Id(p)


# stress tensor
def sigma_cyl(v, p):
    return 2*mu()*epsilon_cyl(v)-Id(p)


def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])


# these should be the same for cylindric and Cartesian... I think ... x.x
def epsilon2d(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


def sigma2d(v, p):
    return 2*mu()*epsilon2d(v) - p*Identity(2)


def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)

Gamma = Constant(10.0)
Pe = Constant(27.0)
Bi = Constant(1.0)
#Bi = Constant(100.0) # Bi large
# Cartesian
u_in = Constant(-2.0)

# Cylindric
u_in_cyl = Constant(-4.0)
#uin_val = "Org"
#uin_val = "Twice"
#uin_val = "Half"
val =  "Bi1" #Thing that I am changing
#uin_val = "Two"
u_c = Constant(-1.0)



# Choose a mesh. You can also import a previously adapted mesh

#domain = Polygon([Point(0.2, 0), Point(0.2, 1), Point(0.1, 1), Point(0, 1), Point(0, 0)])
#mesh = generate_mesh(domain, 8)
#mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 60, 60)
mesh = Mesh('Meshes/CoupledRefinedMeshGamma5Cartesian.xml')


# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
S = FiniteElement("CG", triangle, 1)


inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'

Xvalues = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2]
x_1 = 0.3
x_2 = 0.1
x_min = 0
x_max = 1
tol = 1e-10
x = SpatialCoordinate(mesh)
Qc2 = Expression("Qfun*exp ( -pow( x[1] -(( x1-x2 )/2 + x2), 2 )/( 2*pow( x1-x2,2 ) ) )", degree=3, Qfun=2.3710,
                 x1=0.3,
                 x2=0.1)

vel_flux_values = []
normal_stress_values_ds0 = []
normal_stress_values_ds1 = []
normal_stress_values_ds3 = []
# coord = "Car"
coord = "Cyl"
for i in range(len(Xvalues)):
    print('X value = ', Xvalues[i])
    n = FacetNormal(mesh)

    W = FunctionSpace(mesh, MixedElement([V, Q, S]))

    # Define Function and TestFunction(s)
    w = Function(W)
    (u, p, T) = split(w)
    (v, q, s1) = split(TestFunction(W))


    # Define the viscosity and bcs
    # Qfun should be 2.3710
    Qc2.x1 = x_1 + Xvalues[i]
    Qc2.x2 = x_2 + Xvalues[i]
    x_min = 0 + Xvalues[i]
    x_max = 1 + Xvalues[i]
    Ta = conditional(ge(x[1], x_max), tol,  conditional(ge(x[1], x_min), 1-x[1] + Xvalues[i], 1.0-tol))

    bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
    bcu_inflow_cyl = DirichletBC(W.sub(0), (0.0, u_in_cyl), inflow)
    bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
    bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
    bcP_outflow = DirichletBC(W.sub(1), 0.0, outflow)
    bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
    bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
    bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry, bcT_inflow]
    bcs_cyl = [bcu_inflow_cyl, bcu_wall, bcu_outflow, bcu_symmetry, bcT_inflow]


    facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function

    CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
    CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
    CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
    CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
    CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)

    # Create the measure
    ds = Measure("ds", domain=mesh, subdomain_data=facet_f)

    # Cartesian
    a = (inner(sigma(u, p), epsilon(v)) - div(u) * q + (dot(u, grad(T)) * s1 + (
           1 / Pe) * inner(grad(s1), grad(T)))) * dx() - (1 / Pe) * (-Bi * s1 * T * ds(2))
    b0 = - dot(dot(sigma2d(u, p), v), n) * ds(0) - s1*dot(grad(T), n)*ds(0)
    b2 = - dot(dot(sigma2d(u, p), v), n) * ds(2)
    b3 = -dot(dot(sigma2d(u, p), v), n) * ds(3)
    b4 = + dot(p*n, v) * ds(4)
    L = (- Qc2*s1) * dx() - (1/Pe) * (s1 * Bi * Ta * ds(2))
    F = a + L + b0 + b2 + b3 + b4

    # Cylindric
    a_cyl = (inner(sigma_cyl(u, p), epsilon_cyl(v)) - div_cyl(u) * q + (dot(u, grad(T)) * s1 + (
           1 / Pe) * inner(grad(s1), grad(T)))) * x[0] * dx() - (1 / Pe) * (-Bi * s1 * T * x[0] * ds(2))
    L_cyl = (- Qc2*s1) * x[0] * dx() - (1/Pe) * (s1 * Bi * Ta * x[0] * ds(2))
    b0_cyl = - dot(dot(sigma2d(u, p), v), n) * x[0] * ds(0) - s1*dot(grad(T), n) * x[0] * ds(0)
    b2_cyl = - dot(dot(sigma2d(u, p), v), n) * x[0] * ds(2)
    b3_cyl = -dot(dot(sigma2d(u, p), v), n) * x[0] * ds(3)
    b4_cyl = + dot(p*n, v) * x[0] * ds(4) # imposing symmetry condition

    F_cyl = a_cyl + L_cyl + b0_cyl + b2_cyl + b3_cyl + b4_cyl

    dw = TrialFunction(W)
    J = derivative(F, w, dw)
    J_cyl = derivative(F_cyl, w, dw) # I think this is where I mess up

    for Gamma_val in [1, 10, 15, 20, 23]:
        Gamma.assign(Gamma_val)
        print('Gamma =', Gamma_val)
        # Choose either Cylindrical or Cartesian coordinates. Uncomment the relevant one.
        #problem = NonlinearVariationalProblem(F, w, bcs, J)
        #problem = NonlinearVariationalProblem(F_cyl, w, bcs_cyl, J_cyl)

        #solver = NonlinearVariationalSolver(problem)
        #solver.parameters["newton_solver"]["linear_solver"] = 'umfpack'

        #solver.solve()
        solve(F_cyl == 0, w, bcs_cyl)
        #solve(F == 0, w, bcs)
        # Extract solution
        (u, p, T) = w.split()

    File("Results/velocityCoupledCase1" + coord + "_X" + str(Xvalues[i]) + val + ".pvd") << u
    File("Results/TemperatureCoupledCase1" + coord + "_X" + str(Xvalues[i]) + val + ".pvd") << T
    File("Results/pressureCoupledCase1" + coord + "_X" + str(Xvalues[i]) + val + ".pvd") << p
    Vsig = TensorFunctionSpace(mesh, "DG", degree=1)
    sig = Function(Vsig, name="Stress" + str(Xvalues[i]))
    sig.assign(project(sigma2d(u, p), Vsig))
    File("Results/StressCase1" + coord + "_X" + str(Xvalues[i]) + val + ".pvd") << sig

    W2 = FunctionSpace(mesh, S)
    Pmu = project(mu(), W2)

    File("Results/ViscosityCoupled" + coord +"_X" + str(Xvalues[i]) + val + ".pvd") << Pmu

    #flux = dot(u, n) * dot(u, n) * ds(1)
    flux_cyl = dot(u, n) * dot(u, n) * x[0] * ds(1)
    #vel_flux_values.append(assemble(flux))
    vel_flux_values.append(assemble(flux_cyl))

    # Cartesian
    # area1 = assemble(1.0 * ds(1))
    # normal_stress1 = assemble(inner(sig * n, n) * ds(1)) / area1
    # area0 = assemble(1.0 * ds(0))
    # area3 = assemble(1.0 * ds(3))
    # normal_stress0 = assemble(inner(sig * n, n) * ds(0)) / area0
    # normal_stress3 = assemble(inner(sig * n, n) * ds(3)) / area3

    # Cylindric
    area1 = assemble(1.0 * x[0] * ds(1))
    normal_stress1 = assemble(2*np.pi*inner(sig * n, n) * x[0] * ds(1))
    area0 = assemble(1.0 * x[0] * ds(0))
    area3 = assemble(1.0 * x[0] * ds(3))
    normal_stress0 = assemble(2*np.pi*inner(sig * n, n) * x[0] * ds(0))
    normal_stress3 = assemble(2*np.pi*inner(sig * n, n) * x[0] * ds(3))

    normal_stress_values_ds0.append(normal_stress0)
    normal_stress_values_ds1.append(normal_stress1)
    normal_stress_values_ds3.append(normal_stress3)
    #vel_flux_values.append(assemble(flux_cyl))

fig = plt.figure()
plt.plot(Xvalues, vel_flux_values)
plt.ylabel('Velocity flux')
plt.xlabel('X values')
plt.title('Sweep over X values, Cylindric')
plt.show()
values = np.asarray([Xvalues, vel_flux_values])
np.savetxt("Results/XvsFlux" + coord + val + ".csv", values.T, delimiter='\t')

values = np.asarray([Xvalues, normal_stress_values_ds0, normal_stress_values_ds1, normal_stress_values_ds3])
np.savetxt("Results/XvsAverageNormalStressTop" + coord + val + ".csv", values.T, delimiter='\t')