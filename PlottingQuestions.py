from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# t = np.arange(0.0, 1.0, 0.01)
# s = np.sin(2 * np.pi * t)
# v = [1 + s, t]
# fig = plt.figure()
# plt.contour(v)
# plt.ylabel('volts')
# plt.xlabel('My x label')
# plt.title('a sine wave')
# Fixing random state for reproducibility
# np.random.seed(19680801)
#
# ax2 = fig.add_axes([0.15, 0.1, 0.7, 0.3])
# n, bins, patches = ax2.hist(np.random.randn(1000), 50)
# ax2.set_xlabel('time (s)')

#plt.show()

mesh = RectangleMesh(Point(0, 0), Point(0.2, 1.0), 50, 50)
#mesh = Mesh('Meshes/IsothermalRefinedMesh.xml')
n = FacetNormal(mesh)
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
#f = Expression('exp(-kappa*pow(pi, 2)*t)*sin(pi*k*x[0])', degree=2, kappa=1.0, t=0, k=4)

i = 0
Xvalues = [-0.5]
x_min = 0 + Xvalues[i]
x_max = 1 + Xvalues[i]
tol = 1e-10
delta = Xvalues[i]
x = SpatialCoordinate(mesh)
Ta = conditional(ge(x[1], x_max), tol,  conditional(ge(x[1], x_min), 1-x[1] + Xvalues[i], 1.0-tol))
W2 = FunctionSpace(mesh, Q)
Ta_p = project(Ta, W2)
c = plot(Ta_p)
plt.colorbar(c)
plt.show()


u_fun = Expression(('0.0', 'x[0]<=0.1 ? -uin*x[1]+uout*(x[1]-1) : -uout*exp(x[1]*(x[0]-0.2))'), degree=2, uout = 1, uin = 2 )
W2 = FunctionSpace(mesh, V)
u_e = project(u_fun, W2)

c = plot(u_e)
plt.colorbar(c)
plt.show()

u_fun.uin = 4
u_e = project(u_fun, W2)
c = plot(u_e)
plt.colorbar(c)
plt.show()



f = Expression(('0.0', 'x[0]<=0.1 ? 0.0 : -uout*x[1]*x[1]*exp(x[1]*(x[0]-0.2))'), degree=2, uout = 1 )

f_e = project(f, W2)

g = Expression(('0.0', 'x[0]<=0.1 ? -uin+uout : -uout*(x[0]-0.2)*exp(x[1]*(x[0]-0.2))'), degree=2, uout = 1, uin =2 )

g_e = project(g, W2)

c = plot(u_e)
plt.colorbar(c)
plt.show()

c = plot(f_e)
plt.colorbar(c)
plt.show()

c = plot(g_e)
plt.colorbar(c)
plt.show()
#f = Expression('x[0]<=0.1 ? -uin*x[1]+uout*(x[1]-1) : -uout*exp(x[1]*(x[0]-0.2))', degree=2, uout = 1, uin = 2 )
#f = Expression('-uin*(x[1] - (uout/uin)*x[0]/0.2) *(x[0]<=0.1) - uout*x[0]/0.2)*(x[1]>0)', degree=2, uin=-2, uout=-1)

#print(f)