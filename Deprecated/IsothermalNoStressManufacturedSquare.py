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
mesh = RectangleMesh(Point(0, 0), Point(1.0, 1.0), 4, 4, "crossed")

# Manufactured solution NUMBER 1

x = SpatialCoordinate(mesh)
# Manufactured solution NUMBER 2
u_solns_ex = ['-sin(2*pi*x[1])', 'sin(2*pi*x[0])', '0.0']
#u_solns_ex = ['0.0', '-4*(x[0]+1)*(1-x[0])', '0.0']
#p_solns_ex = ['7*x[1]']
p_solns_ex = ['-cos(2*pi*x[0])*cos(2*pi*x[1])']


mu = Constant(1.0) # constant viscosity

inflow = 'near(x[1], 1.0) && x[0]<=0.5'
right = 'near(x[1], 1.0) && x[0]>=0.5'
wall = 'near(x[0], 1.0)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'

# initialise variables
hvalues = []
errors_u = []
errors_p = []
iterations = 6 # total number of iterations
mesh_type = "crossed"
#mesh_type = "uncrossed"
vtkfile_u = File("Results/IsothermalManufacturedVelocitySquare" + mesh_type + ".pvd")

vtkfile_p = File("Results/IsothermalManufacturedPressureSquare" + mesh_type + ".pvd")

vtkfile_sig = File("Results/IsothermalManufacturedStressSquare" + mesh_type + ".pvd")

for i in range(iterations):

    n = FacetNormal(mesh)

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
    g_ex = Expression(u_solns_ex[2], element=Q, domain=mesh)
    f_ex0 = Expression('2*pi*(cos(2*pi*x[1])*sin(2*pi*x[0]) - 2*pi*sin(2*pi*x[1]) )', degree=2)
    f_ex1 = Expression('2*pi*(2*pi*sin(2*pi*x[0]) + cos(2*pi*x[0])*sin(2*pi*x[1])) ', degree=2)

    right_ex = Expression('cos(2*pi*x[0])', element=Q, domain=mesh)
    #f_ex = Constant((0, -1))
    f_ex = as_vector([f_ex0, f_ex1])
    bcu_inflow = DirichletBC(W.sub(0), u_ex, inflow)
    bcu_wall = DirichletBC(W.sub(0), u_ex, wall)
    bcu_outflow = DirichletBC(W.sub(0), u_ex, outflow)
    #bcu_centre = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre) # this is a SYMMETRY condition, we also need to set u[1].dx(0) == 0
    bcu_centre = DirichletBC(W.sub(0), u_ex, centre)
    bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_centre]

    facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)  # FACET function
    CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.5').mark(facet_f, 0)
    CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.5').mark(facet_f, 1)
    CompiledSubDomain('near(x[0], 1.0)').mark(facet_f, 2)  # wall
    CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
    CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)

    ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
    a1 = (inner(sigma(u, p), epsilon(v))) * dx
    a2 = (- div(u) * q + g_ex * q ) * dx
    a3 = (- dot(f_ex, v)) * dx
    b1 = - dot(dot(sigma(u, p), n), v) * ds(0) - dot(dot(sigma(u, p), n), v) * ds(2) - dot(dot(sigma(u, p), n),
                                                                                               v) * ds(3) \
         - dot(dot(sigma(u, p), n), v) * ds(4) - dot(dot(sigma(u_ex, p_ex), n), v) * ds(1) # we include this as the normal component of the viscous stress tensor is zero at ds(4)
    F = a1 + a2 + a3 + b1
    # Solve problem
    solve(F == 0, w, bcs)
    (u, p) = w.split()
    vtkfile_u << (u, i)
    vtkfile_p << (p, i)
    vertex_values_u_ex = u_ex.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)

    vertex_values_p_ex = p_ex.compute_vertex_values(mesh)
    vertex_values_p = p.compute_vertex_values(mesh)

    import numpy as np
    errors_u.append(np.max(np.abs(vertex_values_u_ex - vertex_values_u)))
    errors_p.append(np.max(np.abs(vertex_values_p_ex - vertex_values_p)))
    #errors_u.append(errornorm(u_ex, u))
    #errors_p.append(errornorm(p_ex, p))
    # M_u = inner((u_ex - u), (u_ex - u)) * dx
    # M_p = (p_ex - p)*(p_ex - p) * dx
    # errors_u.append(assemble(M_u))
    # errors_p.append(assemble(M_p))
    Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
    sig_num = Function(Vsig, name="Stress Numeric")
    sig_num.assign(project(sigma(u, p), Vsig))
    vtkfile_sig << (sig_num, i)
    mesh = refine(mesh)
    #n = (i+1)*4
    #mesh = RectangleMesh(Point(0, 0), Point(1.0, 1.0), n, n)
    x = SpatialCoordinate(mesh)



print('The u errors are', errors_u)
print('The p errors are', errors_p)
values = np.asarray([hvalues, errors_u, errors_p])
np.savetxt("Results/ErrorsIsothermalManufacturedSquare" + mesh_type + ".csv", values.T, delimiter='\t')

sig_ex = Function(Vsig, name="Stress Analytic")
sig_ex.assign(project(sigma(u_ex, p_ex), Vsig))
vtkfile_sig << (sig_ex, i+1)