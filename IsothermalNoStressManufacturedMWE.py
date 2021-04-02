from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-p*Identity(2)

#
# def cond(v):
#     return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
#                           [v[1].dx(0), v[1].dx(1)]]))
#
#
# def sigmabc(v, p):
#     return 2*mu*cond(v) - p*Identity(2)

# Initial coarse mesh
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1.0), 4, 4)

# Manufactured solution NUMBER 1
# solns_ex = ['0.0', '-(1-x[1])*(1-x[1])*(x[0]*x[0]-0.04)', '0.0', '2*(1-x[1])*(x[0]*x[0]-0.04)']
# solns_f = ['0.0', '2*(x[0]*x[0]-0.04) + 2*(1-x[1])*(1-x[1])']

# Manufactured solution NUMBER 2
solns_ex = ['-sin(2*pi*x[1])', 'sin(2*pi*x[0])', '-cos(2*pi*x[0])*cos(2*pi*x[1])', '0.0', 'cos(2*pi*x[0])']
solns_f = ['2*pi*(cos(2*pi*x[1])*sin(2*pi*x[0]) - 2*pi*sin(2*pi*x[1]))', '2*pi*(2*pi*sin(2*pi*x[0]) + cos(2*pi*x[0])*sin(2*pi*x[1]))']


mu = Constant(1.0) # constant viscosity

inflow = 'near(x[1], 1.0) && x[0]<=0.1'
right = 'near(x[1], 1.0) && x[0]>=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'

# initialise variables
hvalues = []
errors_u = []
errors_p = []
iterations = 6 # total number of iterations
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
    u_ex = Expression((solns_ex[0], solns_ex[1]), element=V, domain=mesh)
    p_ex = Expression(solns_ex[2], element=Q, domain=mesh)
    g_ex = Expression(solns_ex[3], element=Q, domain=mesh)
    f_ex = Expression((solns_f[0], solns_f[1]), element=V, domain=mesh)
    right_ex = Expression(solns_ex[4], element = Q, domain=mesh)

    bcu_inflow = DirichletBC(W.sub(0), u_ex, inflow)
    bcu_wall = DirichletBC(W.sub(0), u_ex, wall)
    bcu_outflow = DirichletBC(W.sub(0), u_ex, outflow)
    #bcu_centre = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre) # this is a SYMMETRY condition, we also need to set u[1].dx(0) == 0
    bcu_centre = DirichletBC(W.sub(0), u_ex, centre)
    bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_centre]

    x = SpatialCoordinate(mesh)

    facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)  # FACET function
    CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
    CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
    CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
    CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
    CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)
    #File("Results/facets.pvd") << facet_f
    # Create the measure

    ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
    a1 = (inner(sigma(u, p), epsilon(v))) * dx
    a2 = (- div(u) * q + g_ex * q - dot(f_ex, v)) * dx
    b1 = - dot(dot(sigma(u, p), n), v) * ds(0) - dot(dot(sigma(u, p), n), v) * ds(2) - dot(dot(sigma(u, p), n),
                                                                                               v) * ds(3) \
         - dot(dot(sigma(u, p), n), v) * ds(4) - dot(dot(sigma(u_ex, p_ex), n), v) * ds(1) # we include this as the normal component of the viscous stress tensor is zero at ds(4)
    F = a1 + a2 + b1
    # Solve problem
    solve(F == 0, w, bcs)
    (u, p) = w.split()
    errors_u.append(errornorm(u_ex, u))
    errors_p.append(errornorm(p_ex, p))
    #File("Results/pressureIt" + str(i) + ".pvd") << p
    # W2 = FunctionSpace(mesh, V)
    # f_ex_pr = project(f_ex, W2)
    #File("Results/sourceIt" + str(i) + ".pvd") << f_ex_pr
    mesh = refine(mesh)

print('The u errors are', errors_u)
print('The p errors are', errors_p)


# The stress at ds(1) should be zero.
Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
sig = Function(Vsig, name="Stress")
sig.assign(project(sigma(u, p), Vsig))
area1 = assemble(1.0 * ds(1))
normal_stress1 = assemble(inner(sig * n, n) * ds(1))/area1
print("Normal stress at boundary 1 (should be zero)", normal_stress1)