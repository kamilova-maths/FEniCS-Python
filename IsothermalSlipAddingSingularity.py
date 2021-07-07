from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

u_in = Constant(-2.0)
u_c = Constant(-1.0)


# symmetric gradient
def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))



# stress tensor
def sigma(v, p):
    return 2 * mu * epsilon(v) - p * Identity(2)


mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 80, 80, "crossed")
#mesh = Mesh('Meshes/IsothermalRefinedMesh.xml')

# Create mesh
n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q1 = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q1]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p) = split(w)
(v, q1) = split(TestFunction(W))
# Define the viscosity and bcs

mu = Constant(1.0)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
weird = 'near(x[1], 1.0) && x[0] >=0.1'
#weird = 'near( ( ('+abnd+'-1) /0.1)*(x[0] - 0.2) +' + abnd + '- x[1], 0.0) && x[0]>=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
pinpoint = 'near(x[1], 1.0) && near(x[0], 0.1)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
#bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_slip = DirichletBC(W.sub(0).sub(1), Constant(0.0), weird)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcs = [bcu_inflow, bcu_wall, bcu_slip, bcu_symmetry]
# Define the variational form

f = Constant((0, -1))

colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
# We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 4)
CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 0)
CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.1").mark(colors, 1)
CompiledSubDomain("near(x[0], 0.2)").mark(colors, 2)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 3)  # outflow

x = SpatialCoordinate(mesh)

# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=colors)

a1 = (inner(sigma(u, p), epsilon(v))) * dx
a2 = (- div(u) * q1 - dot(f, v)) * dx
b1 = - dot(dot(sigma(u, p), v), n) * ds(1)
b3 = - dot(dot(sigma(u, p), v), n) * ds(3)
b4 = - dot(dot(sigma(u, p), v), n) * ds(4)
F = a1 + a2

solve(F == 0, w, bcs)
# Extract solution
(u, p) = w.split()

N = 600 # size of the arrays below
x0val_right = np.linspace(0.1, 0.2, N)
x0val_left = x0val_right - 0.1

Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
sig = Function(Vsig, name="Stress")
sig.assign(project(sigma(u, p), Vsig))

area0 = assemble(1.0*ds(0))
print("area at ds0 is", area0)

normal_stress_org = assemble(inner(sig*n, n)*ds(0))

print("Normal stress at boundary 0", normal_stress_org)

def pred_sing(x):
    return -(4/np.pi)*((0.1-x)**(-1))

def pred_sing_int(delta, eps):
    return -(4/np.pi)*(np.log(0.1-delta)-np.log(0.1-eps))



print("Original value of normal stress is", normal_stress_org)

# We define the delta value to use, based on just guesswork x.x
delta_values = np.logspace(-1.1, -6, num=15)
eps = delta_values[-1]
stress_values = []

stress_values_2 = []
for i in range(len(delta_values)):
    delta = delta_values[i]
    x0val_delta = np.linspace(0, 0.1-delta, N)
    stress_top_left_calc = []
    for j in range(len(x0val_delta)):
        stress_top_left_calc.append(sig(x0val_delta[j], 1.0)[3])

    val = stress_top_left_calc[-1]
    x0val_rest = np.linspace(x0val_delta[-1], 0.1, N)
    dx_calc = (0.1 - delta) / (N - 1)
    dx_sing = (x0val_rest[-1] - x0val_rest[0]) / (N - 1)
    C_n = np.abs(val / pred_sing(x0val_rest[0]))

    pred_sing_val = pred_sing(x0val_rest[0:-1])

    stress_average = np.abs((np.sum(stress_top_left_calc) * (x0val_left[-1]-x0val_left[0])) / (N) + (np.sum(pred_sing_val)*(x0val_rest[-2]-x0val_rest[0])) / (N-1))
    stress_values.append(stress_average)
    stress_average = integrate.trapz(stress_top_left_calc, x0val_left) + pred_sing_int(delta, eps)
    stress_values_2.append(stress_average)
# stress_top_left_test = []
# for i in range(len(x0val_left)):
#     stress_top_left_test.append(sig(x0val_left[i], 1.0)[3])


y = np.linspace(0 + 0.00001, 0.1 - 0.00001, 101)
points = [(y_, 1.0) for y_ in y] # 2D points
stress_line = np.array([sig(point)[3] for point in points])

plt.plot(y, stress_line, 'k', linewidth=2)
plt.grid(True)

p_line = np.array([p(point) for point in points])

plt.plot(y, p_line, 'k', linewidth=2)
plt.grid(True)


values = np.asarray([delta_values, stress_values])
np.savetxt("Results/DeltavsAvgStressCase2.csv", values.T, delimiter='\t')

#
values = np.asarray([delta_values, stress_values_2])
np.savetxt("Results/DeltavsIntStressCase2.csv", values.T, delimiter='\t')



#flux = dot(u, n) * dot(u, n) * ds(1)
