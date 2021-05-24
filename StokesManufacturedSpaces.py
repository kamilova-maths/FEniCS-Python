from dolfin import *
import numpy as np
import matplotlib.pyplot as plt


# Define the boundary domains
class Inflow(SubDomain):
    def inside(self, x, on_boundary, mid_local = None):
        if mid_local is None:
           mid_local = mid
        return x[0] <= mid_local - DOLFIN_EPS and near(x[1], 1 )


class Right(SubDomain):
    def inside(self, x, on_boundary, mid_local=None):
        if mid_local is None:
            mid_local = mid
        return x[0] >= mid_local - DOLFIN_EPS and x[1] == near(x[1], 1)


class Wall(SubDomain):
    def inside(self, x, on_boundary, end_local=None):
        if end_local is None:
            end_local = end
        return near(x[0], end)


class Centre(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0 )


class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0)



def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-p*Identity(2)

# Initial coarse mesh
mesh = UnitSquareMesh(2, 2, "crossed")
mid = 0.5
end = 1.0

# Manufactured solution NUMBER 1

x = SpatialCoordinate(mesh)
# Manufactured solution NUMBER 2
u_solns_ex = ['sin(4*pi*x[0])*cos(4*pi*x[1])', '-cos(4*pi*x[0])*sin(4*pi*x[1])']
p_solns_ex = ['pi*cos(4*pi*x[0])*cos(4*pi*x[1])', '-9*pi*cos(4*pi*x[0])']
f_solns_ex = ['28*pi*pi*sin(4*pi*x[0])*cos(4*pi*x[1])', '-36*pi*pi*cos(4*pi*x[0])*sin(4*pi*x[1])']

mu = Constant(1.0) # constant viscosity

# initialise variables
hvalues = []
errors_u = []
errors_p = []
iterations = 7 # total number of iterations
normal_stress_average = []

for i in range(iterations):

    n = FacetNormal(mesh)

    hvalues.append(mesh.hmin())
    #
    # h = 2*Circumradius(mesh)
    # beta = 0.2
    # delta = beta*h**2

    # MINI
    # P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    # B = FiniteElement("Bubble", mesh.ufl_cell(), mesh.topology().dim() + 1)
    # V = VectorElement(NodalEnrichedElement(P1, B))
    # Q = P1
    # W = FunctionSpace(mesh, V * Q)

    # Taylor-Hood
    V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = V * Q
    W = FunctionSpace(mesh, TH)

    # CD
    # V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    # Q = FiniteElement("DG", mesh.ufl_cell(), 0)
    # CD = V * Q
    # W = FunctionSpace(mesh, CD)

    # STAB
    # V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    # Q = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    # ST = V * Q
    # W = FunctionSpace(mesh, ST)
    # Define trial and test functions
    w = Function(W)
    (u, p) = split(w)
    (v, q) = split(TestFunction(W))

    # Manufactured solution of the easy case
    u_ex = Expression((u_solns_ex[0], u_solns_ex[1]), element=V, domain=mesh)
    p_ex = Expression(p_solns_ex[0], element=Q, domain=mesh)
    g_ex = Expression('0.0', element=Q, domain=mesh)
    f_ex = Expression((f_solns_ex[0], f_solns_ex[1]), element=V, domain=mesh)
    right_ex = Expression(p_solns_ex[1], element=Q, domain=mesh)
    inflow = Inflow()
    right = Right()
    wall = Wall()
    centre = Centre()
    outflow = Outflow()
    # Apparently it is more efficient to define them as we have below, as it moves the computation of which nodes
    # belong to the boundary from Python to C++. [But it is important to know the difference in implementation, and the
    # above usage allows me to easily change the shape of the mesh (for example).

    # inflow = 'near(x[1], 1.0) && x[0]<=0.5'
    # right = 'near(x[1], 1.0) && x[0]>=0.5'
    # wall = 'near(x[0], 1.0)'
    # centre = 'near(x[0], 0.0)'
    # outflow = 'near(x[1], 0.0)'
    bcu_inflow = DirichletBC(W.sub(0), u_ex, inflow)
    bcu_wall = DirichletBC(W.sub(0), u_ex, wall)
    bcu_outflow = DirichletBC(W.sub(0), u_ex, outflow)
    bcu_centre = DirichletBC(W.sub(0), u_ex, centre)
    bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_centre]

    facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)  # FACET function
    CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.5').mark(facet_f, 0)
    CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.5').mark(facet_f, 1)
    CompiledSubDomain('near(x[0], 1.0)').mark(facet_f, 2)  # wall
    CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
    CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)
    #File("Results/facets.pvd") << facet_f
    # Create the measure

    ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
    # Define the variational problem
    a1 = (inner(sigma(u, p), epsilon(v))) * dx
    a2 = (- div(u) * q ) * dx
    a3 = (- dot(f_ex, v)) * dx
    b1 = - dot(dot(sigma(u, p), n), v) * ds(0) - dot(dot(sigma(u, p), n), v) * ds(2) - dot(dot(sigma(u, p), n),
                                                                                               v) * ds(3) \
         - dot(right_ex*n, v) * ds(1) #- dot(dot(sigma(u_ex, p_ex), n), v) * ds(1) # we include this as the normal component of the viscous stress tensor is zero at ds(4)
    #a4 = delta*inner(grad(p), grad(q)) * dx
    #a5 = delta*inner(f_ex, grad(q))*dx
    F = a1 + a2 + a3 + b1 # + a4 + a5

    solve(F == 0, w, bcs)
    (u, p) = w.split()
    errors_u.append(errornorm(u_ex, u))
    errors_p.append(errornorm(p_ex, p))
    Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
    sig_num = Function(Vsig, name="Stress Numeric")
    sig_num.assign(project(sigma(u, p), Vsig))
    area1 = assemble(1.0 * ds(1))
    normal_stress_average.append(assemble(inner(sig_num * n, n) * ds(1)) / area1)
    #File("Results/pressureIt" + str(i) + ".pvd") << p
    # W2 = FunctionSpace(mesh, V)
    # f_ex_pr = project(f_ex, W2)
    #File("Results/sourceIt" + str(i) + ".pvd") << f_ex_pr
    mesh = refine(mesh)
    x = SpatialCoordinate(mesh)
print('The u errors are', errors_u)
print('The p errors are', errors_p)

# Saving the errors as mesh is refined

values = np.asarray([hvalues, errors_u, errors_p, normal_stress_average])
np.savetxt("Results/ConvergenceRateTH.csv", values.T, delimiter='\t')

# Post-processing calculations

# Numerical values for the stress tensor
# Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
# sig_num = Function(Vsig, name="Stress Numeric")
# sig_num.assign(project(sigma(u, p), Vsig))
# area0 = assemble(1.0 * ds(0))
# area1 = assemble(1.0 * ds(1))
# area2 = assemble(1.0 * ds(2))
# area3 = assemble(1.0 * ds(3))
# area4 = assemble(1.0 * ds(4))
#
# normal_stress0 = assemble(inner(sig_num * n, n) * ds(0)) / area0
# normal_stress1 = assemble(inner(sig_num * n, n) * ds(1)) / area1
# normal_stress2 = assemble(inner(sig_num * n, n) * ds(2)) / area2
# normal_stress3 = assemble(inner(sig_num * n, n) * ds(3)) / area3
# normal_stress4 = assemble(inner(sig_num * n, n) * ds(4)) / area4
#
# normal_stress_averages = np.asarray([normal_stress0, normal_stress1, normal_stress2, normal_stress3, normal_stress4])
# np.savetxt("Results/normal_stress_average.csv", normal_stress_averages.T, delimiter='\t')
#
# File("Results/normal_stress.pvd") << sig_num
#

# Analytical values for the stress tensor
#
# x1 = np.linspace(0, 1, 200)
#
# stress_top = -9*np.pi*(np.cos(4*np.pi*x1))
#
# values = np.asarray([x1, stress_top])
#
# np.savetxt("Results/stress_top_analytic.csv", values.T, delimiter='\t')
# Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
# sig_ana = Function(Vsig, name="Stress Analytic")
# sig_ana.assign(project(sigma(u_ex, p_ex), Vsig))
#
#
# normal_stress0 = assemble(inner(sig_ana * n, n) * ds(0)) / area0
# normal_stress1 = assemble(inner(sig_ana * n, n) * ds(1)) / area1
# normal_stress2 = assemble(inner(sig_ana * n, n) * ds(2)) / area2
# normal_stress3 = assemble(inner(sig_ana * n, n) * ds(3)) / area3
# normal_stress4 = assemble(inner(sig_ana * n, n) * ds(4)) / area4
#
# normal_stress_averages = np.asarray([normal_stress0, normal_stress1, normal_stress2, normal_stress3, normal_stress4])
# np.savetxt("Results/normal_stress_averages_analytic.csv", normal_stress_averages.T, delimiter='\t')
#
# File("Results/normal_stress_analytic.pvd") << sig_ana


