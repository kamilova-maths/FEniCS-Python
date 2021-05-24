from dolfin import *


# Define the boundary domains
class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-p*Identity(2)


# Mesh
mesh = UnitSquareMesh(50, 50, "crossed")

x = SpatialCoordinate(mesh)

# Manufactured solution
u_solns_ex = ['sin(4*pi*x[0])*cos(4*pi*x[1])', '-cos(4*pi*x[0])*sin(4*pi*x[1])']
p_solns_ex = ['pi*cos(4*pi*x[0])*cos(4*pi*x[1])', '-9*pi*cos(4*pi*x[0])']
f_solns_ex = ['28*pi*pi*sin(4*pi*x[0])*cos(4*pi*x[1])', '-36*pi*pi*cos(4*pi*x[0])*sin(4*pi*x[1])']

mu = Constant(1.0) # constant viscosity
n = FacetNormal(mesh)

#Taylor-Hood
V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = V * Q
W = FunctionSpace(mesh, TH)

# MINI
# P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# B = FiniteElement("Bubble", mesh.ufl_cell(), mesh.topology().dim() + 1)
# V = VectorElement(NodalEnrichedElement(P1, B))
# Q = P1
# W = FunctionSpace(mesh, V * Q)

# CD
# V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
# Q = FiniteElement("DG", mesh.ufl_cell(), 0)
# CD = V * Q
# W = FunctionSpace(mesh, CD)

# Define trial and test functions
w = Function(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Manufactured solution of the easy case
u_ex = Expression((u_solns_ex[0], u_solns_ex[1]), element=V, domain=mesh)
f_ex = Expression((f_solns_ex[0], f_solns_ex[1]), element=V, domain=mesh)


bcs = DirichletBC(W.sub(0), u_ex, Boundary())

# Define the variational problem
a1 = (inner(sigma(u, p), epsilon(v))) * dx
a2 = (- div(u) * q) * dx
a3 = (- dot(f_ex, v)) * dx
F = a1 + a2 + a3

solve(F == 0, w, bcs)
(u, p) = w.split()

# Post-processing calculations
# Numerical values for the stress tensor
Vsig = TensorFunctionSpace(mesh, "DG", degree=0) # copied from texts but not understood
sig_num = Function(Vsig, name="Stress Numeric")
sig_num.assign(project(sigma(u, p), Vsig))
