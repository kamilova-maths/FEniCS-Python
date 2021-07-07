from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-p*Identity(2)


# Initial coarse mesh
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1.0), 10, 10, "crossed")

# We plot the mesh to remind ourselves if it is crossed or uncrossed
plot(mesh)
plt.show()

# We define the expressions for the manufactured solution
# Manufactured solution NUMBER 1
u_solns_ex = ['-sin(2*pi*x[1])', 'sin(2*pi*x[0])', '0.0']
p_solns_ex = ['-cos(2*pi*x[0])*cos(2*pi*x[1])']
f_solns_ex = ['2*pi*(cos(2*pi*x[1])*sin(2*pi*x[0]) - 2*pi*sin(2*pi*x[1]) )',
              '2*pi*(2*pi*sin(2*pi*x[0]) + cos(2*pi*x[0])*sin(2*pi*x[1])) ']


# Manufactured solution NUMBER 2 - this solution does not work because I have not changed f accordingly
# u_solns_ex = ['sin(x[0])*(a*sin(a*x[1]) - cos(a)*sinh(x[1]))', 'cos(x[0])*(a*cos(a*x[1]) + cos(a)*cosh(x[1]))', '0.0']
# p_solns_ex = ['(1+a*a)*cos(a)*cos(x[0])*sinh(x[1])']
# f_solns_ex = ['2*pi*(cos(2*pi*x[1])*sin(2*pi*x[0]) - 2*pi*sin(2*pi*x[1]) )', ]

# Manufactured solution NUMBER 3
# u_solns_ex = ['sin(4*pi*x[0])*cos(4*pi*x[1])', '-cos(4*pi*x[0])*sin(4*pi*x[1])']
# p_solns_ex = ['pi*cos(4*pi*x[0])*cos(4*pi*x[1])', '-9*pi*cos(4*pi*x[0])']
# f_solns_ex = ['28*pi*pi*sin(4*pi*x[0])*cos(4*pi*x[1])', '-36*pi*pi*cos(4*pi*x[0])*sin(4*pi*x[1])']
mu = Constant(1.0) # constant viscosity

# We define the boundaries
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
right = 'near(x[1], 1.0) && x[0]>=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'

# initialise variables
hvalues = []
errors_u_max = []
errors_p_max = []

errors_u = []
errors_p = []
iterations = 5 # total number of iterations
mesh_type = "crossed"
#mesh_type = "uncrossed"


for i in range(iterations):

    n = FacetNormal(mesh)
    # We store the minimum cell size in each iteration for convergence comparison
    hvalues.append(mesh.hmin())

    # Define Taylor--Hood function space W
    V = VectorElement("CG", triangle, 2)  # original spaces
    Q = FiniteElement("CG", triangle, 1)  # original spaces
    W = FunctionSpace(mesh, MixedElement([V, Q]))

    # Define Function and TestFunction(s)
    w = Function(W)
    (u, p) = split(w)
    (v, q) = split(TestFunction(W))

    # Manufactured solution of the easy case
    u_ex = Expression((u_solns_ex[0], u_solns_ex[1]), element=V, domain=mesh)
    p_ex = Expression(p_solns_ex[0], element=Q, domain=mesh)

    f_ex0 = Expression(f_solns_ex[0], degree=2)
    f_ex1 = Expression(f_solns_ex[1], degree=2)

    f_ex = as_vector([f_ex0, f_ex1])

    bcu_inflow = DirichletBC(W.sub(0), u_ex, inflow)
    bcu_wall = DirichletBC(W.sub(0), u_ex, wall)
    bcu_outflow = DirichletBC(W.sub(0), u_ex, outflow)
    bcu_centre = DirichletBC(W.sub(0), u_ex, centre)
    bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_centre]

    facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)  # FACET function
    CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
    CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
    CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
    CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
    CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)

    ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
    a1 = (inner(sigma(u, p), epsilon(v))) * dx
    a2 = (- div(u) * q ) * dx
    a3 = (- dot(f_ex, v)) * dx
    b1 = - dot(dot(sigma(u, p), n), v) * ds(0) - dot(dot(sigma(u, p), n), v) * ds(2) - dot(dot(sigma(u, p), n),
                                                                                               v) * ds(3) \
         - dot(dot(sigma(u, p), n), v) * ds(4) - dot(dot(sigma(u_ex, p_ex), n), v) * ds(1) # we include this as the normal component of the viscous stress tensor is zero at ds(4)
    F = a1 + a2 + a3 + b1
    # Solve problem
    solve(F == 0, w, bcs)
    (u, p) = w.split()

    # Save the results in separate .pvd files for analysis and safekeeping
    File("Results/IsothermalManufacturedVelocity" + mesh_type + str(i) + ".pvd") << u
    File("Results/IsothermalManufacturedPressure" + mesh_type + str(i) + ".pvd") << p

    # We redefine the exact solution at a higher degree of accuracy to obtain a better estimate for the error
    u_ex_fine = Expression((u_solns_ex[0], u_solns_ex[1]), degree=10)
    p_ex_fine = Expression(p_solns_ex[0], degree=10)

    vertex_values_u_ex = u_ex_fine.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)

    vertex_values_p_ex = p_ex_fine.compute_vertex_values(mesh)
    vertex_values_p = p.compute_vertex_values(mesh)

    import numpy as np

    errors_u_max.append(np.max(np.abs(vertex_values_u_ex - vertex_values_u)))
    errors_p_max.append(np.max(np.abs(vertex_values_p_ex - vertex_values_p)))
    errors_u.append(errornorm(u_ex, u, 'H1'))
    errors_p.append(errornorm(p_ex, p, 'L2'))
    # M_u = inner((u_ex - u), (u_ex - u)) * dx
    # M_p = (p_ex - p)*(p_ex - p) * dx
    # errors_u.append(assemble(M_u))
    # errors_p.append(assemble(M_p))
    # We compute the numerical stress expression
    Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
    sig_num = Function(Vsig, name="Stress Numeric")
    sig_num.assign(project(sigma(u, p), Vsig))
    File("Results/IsothermalManufacturedStress" + mesh_type + str(i) + ".pvd") << sig_num
    mesh = refine(mesh)
    x = SpatialCoordinate(mesh)

# We print out the errors for quick inspection

print('The u errors are', errors_u)
print('The p errors are', errors_p)
values = np.asarray([hvalues, errors_u, errors_p])
np.savetxt("Results/L2ErrorsIsothermalManufactured" + mesh_type + ".csv", values.T, delimiter='\t')
print('The maximum u errors are', errors_u_max)
print('The maximum p errors are', errors_p_max)

values = np.asarray([hvalues, errors_u_max, errors_p_max])
np.savetxt("Results/MaxErrorsIsothermalManufactured" + mesh_type + ".csv", values.T, delimiter='\t')

# Extracting the analytic expressions for u_ex and p_ex by projecting onto the appropriate space
W2 = FunctionSpace(mesh, V)
u_ex_proj = project(u_ex, W2)

W2 = FunctionSpace(mesh, Q)
p_ex_proj = project(p_ex, W2)

File("Results/IsothermalManufacturedVelocity" + mesh_type + "Ana" + ".pvd") << u_ex_proj
File("Results/IsothermalManufacturedPressure" + mesh_type + "Ana" + ".pvd") << p_ex_proj

