from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
# Define mesh and geometry - We solve for half of the domain we need, and impose symmetry
#domain = Polygon([Point(1, 0), Point(1, 0.5), Point(0.5, 1), Point(0, 1), Point(0, 0)])
#mesh = generate_mesh(domain, 100)
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1.0), 50, 50)
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

# Define Function and TestFunction(s)
w = Function(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Define the viscosity and bcs

mu = Constant(1.0)
# mu = Expression('exp(-a*pow(x[0],2))', degree=2, a=10)

u_in = Constant(-2.0)
u_c = Constant(-1.0)

# def inflow(x, on_boundary):
#     return on_boundary and near(x[1], 1.0) and x[0] <= 0.5
#
#
# def wall(x, on_boundary):
#     return on_boundary and near(x[0], 1.0)
#
#
# def centre(x, on_boundary):
#     return on_boundary and near(x[0], 0.0)
#
#
# def outflow(x, on_boundary):
#     return on_boundary and near(x[1], 0.0)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<=0.1' # This is C++ code.
right = 'near(x[1], 1.0) && x[0]>=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
#bcP_bottom = DirichletBC(W.sub(1), 0.0, outflow)
#bcP_right = DirichletBC(W.sub(1), 0.0, right)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcP_right, bcu_symmetry]
# Define stress tensor
# epsilon = sym(grad(u))
x = SpatialCoordinate(mesh)


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
f = Constant((0, -10))
# The vectors defined in Fenics are automatically dimensional. We introduce the aspect ratio here, turning the dimensional
# v and u into the non-dimensional ones vsc and usc, noting that usc[0] = (1/asp)*u[0] (or equivalently, we multiply the
# second component by asp.
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
# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
#dS = Measure("dS", domain=mesh, subdomain_data=subdomains)
a1 = (inner(sigma(u, p), epsilon(v))) * dx
a2 = (- div(u) * q - dot(f, v)) * dx
b1 = dot(p*n, v)*ds(0) - dot(mu*dot(nabla_grad(u), n), v)*ds(0) + dot(p*n, v)*ds(2) - dot(mu*dot(nabla_grad(u), n), v)*ds(2) + \
      dot(p*n, v)*ds(3) - dot(mu*dot(nabla_grad(u), n), v)*ds(3) + dot(p*n, v)*ds(4) - dot(mu*dot(nabla_grad(u), n), v)*ds(4)
#b1 = - dot(dot(sigmabc(u, p), n), v)*ds(0) - dot(dot(sigmabc(u, p), n), v)*ds(2) - dot(dot(sigmabc(u, p), n), v)*ds(3) \
     #- dot(dot(sigmabc(u, p), n), v)*ds(4)
# For the boundary terms, note that ds(3) is the only one here that varies along x[1], which is where the asp rescaling
# is. Therefore we have to multiply only that term by (1/asp), or equivalently, multiply the other terms by asp.
# b0 = - dot(dot(sigmabc(u, p), v), n) * ds(0)
# b2 = - dot(dot(sigmabc(u, p), v), n) * ds(2)
# b3 = - dot(dot(sigmabc(u, p), v), n) * ds(3)
F = a1 + a2 + b1
# Solve problem
solve(F == 0, w, bcs)

# Plot solutions
(u, p) = w.split()

# Compute stress tensor
sigma = 2 * mu * grad(u) - p*Identity(len(u))

# Compute surface traction
T = -sigma*n

# Compute normal and tangential components
Tn = inner(T,n) # scalar valued
Tt = T - Tn*n # vector valued

# Piecewise constant test functions
scalar = FunctionSpace(mesh, "DG", 0)
vector = VectorFunctionSpace(mesh, "DG", 0)
v1 = TestFunction(scalar)
w1 = TestFunction(vector)

# Assemble piecewise constant functions for stress
normal_stress = Function(scalar)
shear_stress = Function(vector)

Ln = (1 / FacetArea(mesh))*v1*Tn*ds
Lt = (1 / FacetArea(mesh))*inner(w1, Tt)*ds
assemble(Ln, tensor=normal_stress.vector())
assemble(Lt, tensor=shear_stress.vector())

File("Results/NormalStressIsothermalNoStress.pvd") << normal_stress
# c = plot(normal_stress, title='stress?')
# plt.colorbar(c)
# plt.show()

Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
sig = Function(Vsig, name="Stress")
sig.assign(project(sigmabc(u, p), Vsig))
area0 = assemble(1.0*ds(0))
area1 = assemble(1.0 * ds(1))
area2 = assemble(1.0 * ds(2))
area3 = assemble(1.0 * ds(3))
area4 = assemble(1.0 * ds(4))
print("area at ds0 is", area0)
print("area at ds1 is", area1)
print("area at ds2 is", area2)
print("area at ds3 is", area3)
print("area at ds4 is", area4)
normal_stress0 = assemble(inner(sig*n, n)*ds(0))/area0
normal_stress1 = assemble(inner(sig * n, n) * ds(1))/area1
normal_stress2 = assemble(inner(sig * n, n) * ds(2))/area2
normal_stress3 = assemble(inner(sig * n, n) * ds(3))/area3
normal_stress4 = assemble(inner(sig * n, n) * ds(4))/area4
print("Stress at (0.1, 1):", sig(0.1, 1))
print("Normal stress at boundary 0", normal_stress0)
print("Normal stress at boundary 1", normal_stress1)
print("Normal stress at boundary 2", normal_stress2)
print("Normal stress at boundary 3", normal_stress3)
print("Normal stress at boundary 4", normal_stress4)


# File("Results/normalstressNoStress.pvd") << normal_stress
# File("Results/shearstressNoStress.pvd") << shear_stress
# dx = 1/(len(normal_stress.vector().get_local())-1)

# print(" I think the answer of this is", np.cumsum(normal_stress.vector().get_local()))
#
# print("The length of this vector is", len(normal_stress.vector().get_local()))
#
# print("I think all the other elements are going to be zero, for example", normal_stress.vector().get_local())

# We check stress at the end
# Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
# sig = Function(Vsig, name="Stress")
# sig.assign(project(sigmabc(u, p), Vsig))
#n3 = as_vector([n[0], n[1], 0])
#stress = (sigmabc(u, p)[0, 0]*n[0] + sigmabc(u, p)[1, 1]*n[1])*ds(1)
#stress = dot(dot(sigmabc(u, p), u), n) * ds(1) # highly unlikely this is the right way to compute normal stress at ds2
# parameters['ghost_mode'] = 'none' #  'shared_facet'
#stress1 = (dot(as_vector([sigmabc(u, p)[0, 0], sigmabc(u, p)[1, 1]]), n('-'))) * dS(domain=mesh, subdomain_data=colors, subdomain_id=0)
# stress2 = dot(dot(sigmabc(u, p)('-'), as_vector([1.0, 1.0])), n('+') ) * dS(domain=mesh, subdomain_data=subdomains, subdomain_id = 1) # This is some weird stuff here, WHY! x.x
# stresss0= dot(as_vector([0.0, 3.0]), n) * ds(0)
# stresss1= dot(as_vector([0.0, 3.0]), n) * ds(1)
# stresss2= dot(as_vector([0.0, 3.0]), n) * ds(2)
# stresss3= dot(as_vector([0.0, 3.0]), n) * ds(3)
# stresss4= dot(as_vector([1.0, 0.0]), n) * ds(4)
# print("Normal at Omega0, should be 1.5", assemble(stresss0))
# print("Normal at Omega1, should be 1.5, but is probably weird AF", assemble(stresss1))
# print("Normal at Omega2, should be 1.0", assemble(stresss2))
# print("Normal at Omega3, should be -3.0", assemble(stresss3))
# print("Normal at Omega4, should be -1.0", assemble(stresss4))
# stresss0= dot(as_vector([1.0, 2.0]), n)
# stresss1= dot(as_vector([1.0, 2.0]), n)
# stresss2= dot(as_vector([1.0, 2.0]), n)
# stresss3= dot(as_vector([1.0, 2.0]), n)
# stresss4= dot(as_vector([1.0, 2.0]), n)
# print("Normal at Omega0, should be -2.0", stresss0)
# print("Normal at Omega1, should be -2.0, but is probably weird AF", stresss1)
# print("Normal at Omega2, should be 1.0", stresss2)
# print("Normal at Omega3, should be -2.0",stresss3)
# print("Normal at Omega4, should be 1.0", stresss4)
# ntr = as_vector([n[1], 0.0])
# stress2 = dot(dot(sigmabc(u, p), n) , n) * ds(1) # highly unlikely this is the right way to compute normal stress at ds2
# stress3 = dot(as_vector([sigmabc(u, p)[0, 0]*n[0], sigmabc(u, p)[1, 1]*n[1]]), n) * ds(1)
# stress4 = (dot(as_vector([sigmabc(u, p)[0, 0], sigmabc(u, p)[0, 1]]), n) + dot(as_vector([sigmabc(u, p)[1, 0], sigmabc(u, p)[1, 1]]), n)) * ds(1)
# #sigmaTrans = as_tensor([sigmabc(u, p).T])
# #stress5 = dot((sigmabc(u, p), n), n) * ds(1)
# #stress5 =( sigmabc(u, p)[0, 1] + sigmabc(u, p)[1, 0]) * ds(1)
# #bmesh = BoundaryMesh(mesh, 'exterior')
# #n2 = get_facet_normal(bmesh)
#
# #fid = File('normal.pvd')
# #fid << n2
flux = dot(u, n) * dot(u, n) * ds(1)
# #stress2 = sigmabc(u, p)[0, 0] * ds(3)
# #stress = 2*((dot(as_vector([u[0].dx(0), 0.5*(u[1].dx(0))]), n) ) * (dot(as_vector([u[0].dx(0), 0.5*(u[1].dx(0))]), n) )) * ds(2)
# #stress = 2*dot(as_vector([u[0].dx(0), 0.5*(u[0].dx(1) + u[1].dx(0))]), n) * dot(as_vector([u[0].dx(0), 0.5*(u[0].dx(1) + u[1].dx(0))]), n) * ds(2)
# #stress = 2*dot(as_vector([u[0].dx(0), 0.5*(u[0].dx(1) + u[1].dx(0))]), n) * ds(1)
# #total_stress1 = assemble(stress1)
# total_flux_new = assemble(flux)
# #print("Total stress1 is", total_stress1)
# total_stress2 = assemble(stress2)
# print("Total stress2 is", total_stress2)
# total_stress3 = assemble(stress3)
# print("Total stress3 is", total_stress3)
# total_stress4 = assemble(stress4)
# print("Total stress4 is", total_stress4)

#print("Total stress 5 is", assemble(stress5))
# print("Total flux on boundary 1 is", total_flux_new)
#
#
# #total_stress_new = assemble(stress2)
# #print("Total stress on boundary 2 is", total_stress_new)
File("Results/velocityIsothermalNoStressDirty.pvd") << u
File("Results/pressureIsothermalNoStressDirty.pvd") << p
# c = plot(u, title='velocity')
# plt.colorbar(c)
# plt.show()