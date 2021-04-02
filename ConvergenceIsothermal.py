from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from ufl import replace
# Define mesh and geometry - We solve for half of the domain we need, and impose symmetry
#domain = Polygon([Point(1, 0), Point(1, 0.5), Point(0.5, 1), Point(0, 1), Point(0, 0)])
#mesh = generate_mesh(domain, 10)
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1.0), 3, 3)
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
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1), 0],
                          [v[1].dx(0), v[1].dx(1), 0],
                          [0, 0, 0]]))


# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-Id(p)


def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])


def cond(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


def sigmabc(v, p):
    return 2*mu*cond(v) - p*Identity(2)


# Define the variational form
f = Constant((0, -1))

facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function
#We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)
CompiledSubDomain('near(x[1], 1.0) && x[0]>0.1 && x[0]<0.2').mark(facet_f, 5)

# Create the measure
x = SpatialCoordinate(mesh)
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
a1 = (inner(sigma(u, p), epsilon(v))) * dx
a2 = (- div(u) * q - dot(f, v)) * dx

F = a1 + a2
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
for i in range(6):
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
    ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
    a1 = (inner(sigma(u, p), epsilon(v))) * dx
    a2 = (- div(u) * q - dot(f, v)) * dx
    F = a1 + a2
    # Solve problem
    solve(F == 0, w, bcs)
    # Plot solutions
    (u, p) = w.split()
    #u_next = interpolate(u, V2)
    #e_prev = np.sqrt(assemble(inner(u_next-u_prev, u_next-u_prev)*dx))
    #errors.append(e_prev)
    p_avg_new = assemble(p * dx) / assemble(1.0 * dx(domain=mesh))
    #p_next = replace(p, {p: p-p_avg})
    #p_next = interpolate(p-p_avg, Q2)
    #p_avg = interpolate(p_avg, Q2)
    #print(p_avg)
   # p_next = p_next - p_avg

    errors_u.append(errornorm(u, u_prev, norm_type='L2'))
    errors_p.append(np.abs(errornorm(p, p_prev, norm_type='L2')))
    errors_p2.append(np.abs(p_avg_new-p_avg_old))
    # plot(mesh)
    # plt.show()


#File("Results/velocityIsothermalMeshRefinement.pvd") << u
#File("Results/pressureIsothermalMeshRefinement.pvd") << p
#c = plot(u, title='velocity')
#plt.colorbar(c)
#plt.show()

rvalues = [0]
for i in range(len(errors_u)-1):
    rvalues.append(np.log(errors_u[i+1]/errors_u[i])/np.log(hvalues[i+1]/hvalues[i]))
print(rvalues)

values = np.asarray([hvalues, errors_u, rvalues])
np.savetxt("Results/ErrorsVelocityIsothermal.csv", values.T, delimiter='\t')

rvalues = [0]
for i in range(len(errors_p)-1):
    rvalues.append(np.log(errors_p[i+1]/errors_p[i])/np.log(hvalues[i+1]/hvalues[i]))
print(rvalues)

values = np.asarray([hvalues, errors_p, rvalues])
np.savetxt("Results/ErrorsPressureIsothermal.csv", values.T, delimiter='\t')

values = np.asarray([hvalues, errors_p2, rvalues])
np.savetxt("Results/ErrorsPressure2Isothermal.csv", values.T, delimiter='\t')
