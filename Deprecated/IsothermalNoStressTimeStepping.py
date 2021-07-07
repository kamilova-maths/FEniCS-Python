from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
# Define mesh and geometry - We solve for half of the domain we need, and impose symmetry

mesh = RectangleMesh(Point(0, 0), Point(0.2, 1.0), 50, 50, "crossed")
#mesh = Mesh('Meshes/IsothermalRefinedMesh.xml')
n = FacetNormal(mesh)

# V = VectorElement("CG", mesh.ufl_cell(),  2)
# Q = FiniteElement("DG", mesh.ufl_cell(), 0)
# P2 = FiniteElement("CG", mesh.ufl_cell(), 2)
# B = FiniteElement("Bubble", mesh.ufl_cell(), mesh.topology().dim() + 1)
# V = VectorElement(NodalEnrichedElement(P2, B))
# Q = FiniteElement("DG", mesh.ufl_cell(), 1)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2) # original spaces
Q = FiniteElement("CG", triangle, 1) # original spaces
W = FunctionSpace(mesh, MixedElement([V, Q]))




def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-p*Identity(2)



# Define the variational form
f = Constant((0, -1))

# Define Function and TestFunction(s)
w = Function(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Define the viscosity and bcs

mu = Constant(1.0)
# mu = Expression('exp(-a*pow(x[0],2))', degree=2, a=10)
inflow = 'near(x[1], 1.0) && x[0]<=0.1' # This is C++ code.
right = 'near(x[1], 1.0) && x[0]>=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
pinpoint = 'near(x[1], 1.0) && near(x[0], 0.1)'


x = SpatialCoordinate(mesh)

u_c = Expression(('0.0','-tanh(t)'), degree=2, t=0)
u_in = Expression(('0.0','-2*tanh(t)'), degree=2, t=0)
# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
u_0 = Expression(('0.0', '0.0'), degree=2)

u_prev = interpolate(u_0, FunctionSpace(mesh, V))
bcu_inflow = DirichletBC(W.sub(0),u_in, inflow)
bcu_wall = DirichletBC(W.sub(0), u_c, wall)
bcu_outflow = DirichletBC(W.sub(0), u_c, outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]
# Define stress tensor

facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function
CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)
# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
a1 = (inner(sigma(u, p), epsilon(v))) * dx
a2 = (- div(u) * q - dot(f, v)) * dx
b1 = - dot(dot(sigma(u, p), n), v)*ds(0) - dot(dot(sigma(u, p), n), v)*ds(2) - dot(dot(sigma(u, p), n), v)*ds(3)

N = 10
T = 0.1 # final fake time [ apparently this is real time x.x]
dt = T/N


F = a1 + a2 + b1 + (1/dt) * dot(u_prev, v)*dx

dw = TrialFunction(W)
##J = derivative(F, w, dw)

vtkfile_u = File('Results/Regularised_u.pvd')

vtkfile_p = File('Results/Regularised_p.pvd')

t = 0
for i in range(N):
    # Update the boundary condition
    t += dt
    u_in.t = t
    u_c.t = t

    # Solve problem
    F = a1 + a2 + b1 + (1 / dt) * dot(u_prev, v) * dx
    J = derivative(F, w, dw)
    problem = NonlinearVariationalProblem(F, w, bcs, J)

    solver = NonlinearVariationalSolver(problem)
    #solver.parameters["newton_solver"]["linear_solver"] = 'umfpack'
   # solver.parameters["nonlinear_solver"]["preconditioner"] = 'amg'

    solver.solve()
   # solve(F == 0, w, bcs)

    # Plot solutions
    (u, p) = w.split()
    vtkfile_u << (u, t)
    vtkfile_p << (p, t)
    u_prev = u


sav = 0.0

if sav == 1.0:
    x0val_right = np.linspace(0.1, 0.2, 100)
    x0val_left = x0val_right - 0.1
    pds0 = []
    pds1 = []

    for i in range(len(x0val_right)):
        pds0.append(p(x0val_left[i], 1.0))
        pds1.append(p(x0val_right[i], 1.0))

    values = np.asarray([x0val_left, pds0, x0val_right, pds1])
    np.savetxt("Results/IsothermalZeroStressPTopuinSmall.csv", values.T, delimiter='\t')

# Don't call this indicator plt ... for obvious reasons Ali
plt_ind = 0.0

if plt_ind == 1.0:
    c = plot(u, title='velocity')
    plt.colorbar(c)
    plt.show()

    c = plot(p, title='pressure')
    plt.colorbar(c)
    plt.show()



