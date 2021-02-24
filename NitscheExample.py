from fenics import *
from dolfin import *
from dolfin_dg import StokesNitscheBoundary, tangential_proj
# Geometry
mesh = UnitSquareMesh(32, 32)
n = FacetNormal(mesh)
# Function space
We = MixedElement([VectorElement("CG", mesh.ufl_cell(), 2),
FiniteElement("CG", mesh.ufl_cell(), 1)])
W = FunctionSpace(mesh, We)
# Manufactured solution
u_soln = Expression(("2*x[1]*(1.0 - x[0]*x[0])","-2*x[0]*(1.0 - x[1]*x[1])"),degree=4, domain=mesh)
p_soln = Constant(0.0)
# Construct an initial guess with no singularity in eta(u)
U = interpolate(Expression(("x[1]", "x[0]", "0.0"), degree=1), W)
# Viscosity model
def eta(u):
    return 1 + sqrt(inner(grad(u), grad(u)))**-1


# Viscous flux operator
def F_v(u, grad_u, p_local=None):
    if p_local is None:
        p_local = p
    return 2 * eta(u) * sym(grad_u) - p_local * Identity(2)
# Variational formulation
u, p = split(U)
v, q = split(TestFunction(W))
f = -div(F_v(u_soln, grad(u_soln), p_soln))
g_tau = tangential_proj(F_v(u_soln, grad(u_soln), p_soln) * n, n)
N = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx \
+ div(u) * q * dx
# Slip boundary conditions
stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
N += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds)
solve(N == 0, U)