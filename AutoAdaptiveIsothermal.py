# I THINK IT WORKS NOW

from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
#mesh = UnitSquareMesh(8, 8)
#domain = Polygon([Point(0.2, 0), Point(0.2, 1), Point(0.1, 1), Point(0, 1), Point(0, 0)])
#mesh = generate_mesh(domain, 30)
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 40, 40)
n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define Function and TestFunction(s)
w = TrialFunction(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Define the viscosity and bcs

mu = Constant(1.0)
#mu = Expression('exp(-a*pow(x[0],2))', degree=2, a=10)

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
x = SpatialCoordinate(mesh)


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


def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)


# Define the variational form
f = Constant((0, -1))
#
#
# class Omega0(SubDomain):
#     def inside(self, x, on_boundary):
#         #return True if x[1] - 1.0 < DOLFIN_EPS and x[0] <= 0.5 else False
#         return True if near(x[1], 1.0) and x[0] <= 0.1 else False
#
#
# class Omega1(SubDomain):
#     def inside(self, x, on_boundary):
#         #return True if x[1] - 1.0 < DOLFIN_EPS and x[0] >= 0.5 else False
#         return True if near(x[1], 1.0) and x[0] >= 0.1 else False
#
# class Omega2(SubDomain):
#     def inside(self, x, on_boundary):
#       #  return True if x[0] - 1.0 < DOLFIN_EPS else False
#         return True if near(x[0], 0.2) else False
#
# class Omega3(SubDomain):
#     def inside(self, x, on_boundary):
#        # return True if x[1] < DOLFIN_EPS else False
#         return True if near(x[1], 0.0) else False
# class Omega4(SubDomain):
#     def inside(self, x, on_boundary):
#         #return True if near(x[0], 0.0) else False
#         return True if near(x[0], 0.0) else False
#
#
# subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0) # CELL function
#
# # Mark subdomains with numbers 1 to 5
# subdomain0 = Omega0()
# subdomain0.mark(subdomains, 0)
#
# # Mark subdomains with numbers 1 to 5
# subdomain1 = Omega1()
# subdomain1.mark(subdomains, 1)
#
# # Mark subdomains with numbers 1 to 5
# subdomain2 = Omega2()
# subdomain2.mark(subdomains, 2)
#
# # Mark subdomains with numbers 1 to 5
# subdomain3 = Omega3()
# subdomain3.mark(subdomains, 3)
#
# # Mark subdomains with numbers 1 to 5
# subdomain4 = Omega4()
# subdomain4.mark(subdomains, 4)

facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function
#colors.set_all(0)  # default to zero
#We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)
CompiledSubDomain('near(x[1], 1.0) && x[0]>0.1 && x[0]<0.2').mark(facet_f, 5)
# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
#dS = Measure("dS", domain=mesh, subdomain_data=subdomains)


a = (inner(sigma(u, p), epsilon(v)) - div(u) * q) * dx()
L = ( - dot(f, v)) * dx()

a_cyl = (inner(sigma(u, p), epsilon_cyl(v)) - div_cyl(u)*q) * x[0] * dx()
L_cyl = (- dot(f, v)) * x[0] * dx()
#a = inner(grad(u), grad(v))*dx()
#L = f*v*dx() + g*v*ds()
# Define function for the solution
w = Function(W)

# Define goal functional (quantity of interest)
M = inner(w[0], w[0])*dx()
M_cyl = inner(w[0], w[0])*x[0]*dx()
# Define error tolerance
tol = 1.e-8

# Solve equation a = L with respect to u and the given boundary
# conditions, such that the estimated error (measured in M) is less
# than tol
problem = LinearVariationalProblem(a, L, w, bcs)
#problem_cyl = LinearVariationalProblem(a_cyl, L_cyl, w, bcs)
solver = AdaptiveLinearVariationalSolver(problem, M)
solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "umfpack"
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
solver.solve(tol)

#solver = LinearVariationalSolver(problem)
#solver.parameters["linear_solver"] = "umfpack"
#solver.solve()
(u, p) = w.split()

#solver.summary()
#mesh0 = mesh.root_node()
#mesh1 = mesh.leaf_node()
# Plot solution(s)

# Plot solutions
(u0, p0) = w.root_node().split()
(u1, p1) = w.leaf_node().split()
#
R = w.leaf_node().function_space()
# plot(R.mesh())
#
#
#File("Results/velocityIsothermalCylindric.pvd") << u
#
# File("Results/velocityIsothermalFinalMesh.pvd") << u1
#
#
# File("Results/pressureIsothermalInitialMesh.pvd") << p0
#
#
# #mesh = Mesh()
# #plot(mesh)
# #plt.show()
# #filename = "output/" + filename + ".xdmf"
# #f = XDMFFile(MPI.comm_world, "IsothermalRefinedMesh.xdmf")
#
File("Meshes/IsothermalAdaptiveMesh.xml") << R.mesh()
#
# plot(u, title="Velocity on initial mesh")
# plt.show()
# plot(u1, title="Velocity on final mesh")
# plt.show()
#
# plot(p0, title="Pressure on initial mesh")
# plt.show()
# plot(p1, title="Pressure on final mesh")
# plt.show()