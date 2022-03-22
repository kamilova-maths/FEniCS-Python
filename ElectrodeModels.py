from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# Define mesh and geometry - 
mesh = Mesh('Meshes/IsothermalAdaptiveMesh.xml') # Mesh file can be found in this path. 

# Define normal vector and spatial coordinate form the mesh
n = FacetNormal(mesh)
x = SpatialCoordinate(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
S = FiniteElement("CG", triangle, 1) # incorporated the finite element for temperature 
W = FunctionSpace(mesh, MixedElement([V, Q, S])) # extended the function space with S

# Define Function and TestFunction(s)
w = Function(W)

(v, q, s1) = split(TestFunction(W)) # We add "s1" for the temperature. The notation is just to separate it from the measure "ds"

dw = TrialFunction(W)  # Define the trial function, for the Jacobian

# Define model parameters 
Gamma = Constant(1.0) # How strongly is the viscosity coupled to the temperature
Pe = Constant(27) # ratio between heat by convection (vel.of paste) and conduction (mat. proper. of paste)
Bi = Constant(11.6) # ratio between heat transfer from the casing and conduction (mat. proper. of the paste)

Q_I = Constant(2.3710) # magnitude of the maximum induction heating 
x1 = Constant(0.7) # position where induction heating begins to increase
x2 = Constant(0.9) # position where induction heating begins to decrease

Delta_T = Constant(260)
T_top = Constant(353)
T_in = Constant(343)
T_bot = Constant(393)


uinval = Constant(4.0) # inlet velocity
ucval = Constant(1.0) # casing velocity

uin = Expression(('0.0', 'uinval'), uinval=uinval, element=V, domain=mesh) # inlet velocity vector 
uc = Expression(('0.0', 'ucval'), ucval=ucval, element=V, domain=mesh) # casing velocity vector

Qc2 = Expression("Q_I*exp ( -pow( x[1] -(( x1+x2 )/2 ), 2 )/( 2*pow( x2-x1,2 ) ) )", element=S, domain=mesh, Q_I=Q_I, x1=x1,
                x2=x2) # induction heating function 

Ta = Expression("(1/Delta_T)*(T_top - T_bot)*x[1]+ (1/Delta_T)*(T_top - T_in)", Delta_T = Delta_T, T_top = T_top, T_in = T_in, T_bot = T_bot, 
                element=S, domain=mesh) # wall heating function 


tol = 1E-14 # we set a tolerance which is zero to computer precision. 


def inflow(x, on_boundary): # this is omega 0
    return on_boundary and abs(x[1]) < tol and x[0] <= 0.1


def free_surface(x, on_boundary): # this is omega 1
    return on_boundary and abs(x[1]) < tol and x[0] >= 0.1


def wall(x, on_boundary): # this is omega 2
    return on_boundary and abs(x[0]-0.2) < tol


def outflow(x, on_boundary): # this is omega 3
    return on_boundary and abs(x[1]-1) < tol


def centre(x, on_boundary): # this is omega 4
    return on_boundary and abs(x[0]) < tol


bcu_inflow = DirichletBC(W.sub(0), uin, inflow)
bcu_wall = DirichletBC(W.sub(0), uc, wall)
bcu_outflow = DirichletBC(W.sub(0), uc, outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)

bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)

bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry, bcT_inflow]


class GoverningEquations(NonlinearProblem):
    def __init__(self, a, L, bcs):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = bcs

    def F(self, b, x):
        assemble(self.L, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.a, tensor=A)
        for bc in self.bcs:
            bc.apply(A)
            
# Form compiler options
# parameters["form_compiler"]["optimize"] = True
# parameters["form_compiler"]["cpp_optimize"] = True


def mu():
    return exp(-Gamma*T)


def epsilon(v): # symmetric gradient in 2D (used for boundaries)
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# 2Dstress tensor (used for boundaries)
def sigma(v, p):
    return 2*mu()*epsilon(v)-p*Identity(2)


# symmetric cylindric gradient 
def epsilon_cyl(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))


# 3D cylindrical stress tensor 
def sigma_cyl(v, p):
    return 2 * mu() * epsilon_cyl(v) - p * Identity(3)


# cylindrical divergence 
def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)


# Define the subdomains for the measure
facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # 

CompiledSubDomain('near(x[1], 0.0) && x[0]<=0.1').mark(facet_f, 0)
CompiledSubDomain('near(x[1], 0.0) && x[0]>=0.1').mark(facet_f, 1)
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 1.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)


# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)

# Define the solution function 
w = Function(W, name="Variables")

u, p, T = split(w)

# Assemble the area and line integrals 
L_flow = (inner(sigma_cyl(u, p), epsilon_cyl(v))) * x[0] * dx() - div_cyl(u) * q  * x[0] * dx()

L_Temp = (dot(u, grad(T))* s1 + (1 / Pe) * inner(grad(s1), grad(T))) * x[0] * dx() - Qc2*s1 * x[0] * dx()
b_Temp = - (1 / Pe) * (-Bi * s1 * T * x[0] * ds(2)-Bi * s1 * T * x[0] * ds(1)) - (1/Pe) * (s1 * Bi * Ta * x[0] * ds(2))

L = L_flow + L_Temp + b_Temp

# Compute the directional derivative

a = derivative(L, w, dw)

problem = GoverningEquations(a, L, bcs)

solver = NewtonSolver()
# solver.parameters["linear_solver"] = "lu"
# solver.parameters["convergence_criterion"] = "incremental"
# solver.parameters["relative_tolerance"] = 1e-6

# Solve problem
solver.solve(problem, w.vector())

# Split the solution into u, p, and T
(u, p, T) = w.split()


# Plot in FEniCS 

# Plot velocity
c = plot(u)
plt.colorbar(c)
plt.show()

# Plot pressure
c = plot(p)
plt.colorbar(c)
plt.show()


# Plot temperature
c = plot(T)
plt.colorbar(c)
plt.show()

# Saving files for visualising with ParaView

File("Results/ElectrodeModel1/PasteFlowCoupledTemp/Gamma" + "{:.0f}".format(float(Gamma)) + "/velocityCylindrical.pvd") << u
File("Results/ElectrodeModel1/PasteFlowCoupledTemp/Gamma" + "{:.0f}".format(float(Gamma)) + "/pressureCylindrical.pvd") << p
File("Results/ElectrodeModel1/PasteFlowCoupledTemp/Gamma" + "{:.0f}".format(float(Gamma)) + "/TemperatureCylindrical.pvd") << T