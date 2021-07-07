from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from ufl import replace
# Define mesh and geometry - We solve for half of the domain we need, and impose symmetry
#domain = Polygon([Point(1, 0), Point(1, 0.5), Point(0.5, 1), Point(0, 1), Point(0, 0)])
#mesh = generate_mesh(domain, 10)
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1.0), 3, 3, "crossed")
#mesh = Mesh('Meshes/IsothermalRefinedMesh.xml')
n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Define the viscosity and bcs

mu = Constant(1.0)
# mu = Expression('exp(-a*pow(x[0],2))', degree=2, a=10)

u_in = Constant(-2.0)
u_c = Constant(-1.0)

inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]


def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


def sigma(v, p):
    return 2*mu*epsilon(v) - p*Identity(2)


# Define the variational form
f = Constant((0, -1))

facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function
#We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)

# Create the measure
x = SpatialCoordinate(mesh)
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
a1 = (inner(sigma(u, p), epsilon(v))) * dx
a2 = (- div(u) * q - dot(f, v)) * dx
b1 = - dot(dot(sigma(u, p), n), v)*ds(0) - dot(dot(sigma(u, p), n), v)*ds(2) - dot(dot(sigma(u, p), n), v)*ds(3)
F = a1 + a2 + b1
# Solve problem
solve(F == 0, w, bcs)

# Plot solutions
(u, p) = w.split()
h_prev = mesh.hmin()
hvalues = [h_prev]
p_avg_old = assemble(p * dx) / assemble(1.0 * dx(domain=mesh))
errors_u = [1]
errors_p = [1]
errors_p2 = [1]

vtkfile_u = File('Results/Isothermal_meshref_u.pvd')

vtkfile_p = File('Results/Isothermal_meshref_p.pvd')

vtkfile_stress = File('Results/Isothermal_meshref_stress.pvd')

Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
#sig = TensorFunctionSpace(mesh, "DG", degree=0)
sig_num = Function(Vsig, name="Stress Numeric")
sig_num.assign(project(sigma(u, p), Vsig))
vtkfile_stress << (sig_num, 0)
normal_stress0 = [assemble(inner(sig_num * n, n) * ds(0)) ]
for i in range(5):
    # Create the measure
    mesh = refine(mesh)
    n = FacetNormal(mesh)
    hvalues.append(mesh.hmin())
    V2 = FunctionSpace(mesh, V)
    Q2 = FunctionSpace(mesh, Q)
    u_prev = interpolate(u, V2) # interpolating u on mesh i + 1
    p_prev = interpolate(p, Q2)
    V = VectorElement("CG", triangle, 2)
    Q = FiniteElement("CG", triangle, 1)
    W = FunctionSpace(mesh, MixedElement([V, Q]))
    w = Function(W)
    (u, p) = split(w)
    (v, q) = split(TestFunction(W))

    bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
    bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
    bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
    bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
    bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]

    x = SpatialCoordinate(mesh)

    facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)  # FACET function
    CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
    CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
    CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
    CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
    CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)
    ds = Measure("ds", domain=mesh, subdomain_data=facet_f)

    a1 = (inner(sigma(u, p), epsilon(v))) * dx
    a2 = (- div(u) * q - dot(f, v)) * dx
    b1 = - dot(dot(sigma(u, p), n), v) * ds(0) - dot(dot(sigma(u, p), n), v) * ds(2) - dot(dot(sigma(u, p), n), v) * ds(
        3)
    F = a1 + a2 + b1
    # Solve problem
    solve(F == 0, w, bcs)
    # Plot solutions
    (u, p) = w.split()

    vtkfile_u << (u, i+1)
    vtkfile_p << (p, i+1)
    Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
    sig_num = Function(Vsig, name="Stress Numeric")
    sig_num.assign(project(sigma(u, p), Vsig))
    vtkfile_stress << (sig_num, i+1)

    area1 = assemble(1.0 * ds(1))
    normal_stress0.append(assemble(inner(sig_num * n, n) * ds(0)) )

    errors_u.append(errornorm(u, u_prev, norm_type='L2'))
    errors_p.append(np.abs(errornorm(p, p_prev, norm_type='L2')))


normal_stress_averages = np.asarray([hvalues, normal_stress0])
np.savetxt("Results/AverageNormalStressds0.csv", normal_stress_averages.T, delimiter='\t')
# rvalues = [0]
# for i in range(len(errors_u)-1):
#     rvalues.append(np.log(errors_u[i+1]/errors_u[i])/np.log(hvalues[i+1]/hvalues[i]))
# print(rvalues)
#
# values = np.asarray([hvalues, errors_u, rvalues])
# np.savetxt("Results/ErrorsVelocityIsothermal.csv", values.T, delimiter='\t')
#
# rvalues = [0]
# for i in range(len(errors_p)-1):
#     rvalues.append(np.log(errors_p[i+1]/errors_p[i])/np.log(hvalues[i+1]/hvalues[i]))
# print(rvalues)
#
# values = np.asarray([hvalues, errors_p, rvalues])
# np.savetxt("Results/ErrorsPressureIsothermal.csv", values.T, delimiter='\t')


