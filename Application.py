import HZ_model_93 as hz93
import numpy as np
import matplotlib.pyplot as plt

#data

r = [0.2,0.4,0.6,1]
v = [0.3,0.3,0.3,0.3]
e = [10,100,1000,10000]
muphase = hz93.mu(e,v)
kphase = hz93.k(e,v)

#application

x = hz93.keff(2,kphase,muphase,r)
y = hz93.mueff(2,muphase,v,r)

print("kphase =", kphase)
print("mphase =", muphase)
print("keff-herve93 =",x,"MPa")
print("mueff-herve93 =",y,"MPa")

#3D graph
X = np.arange(0.01, 10, 0.1)
Y = np.arange(0.01, 10, 0.1)
X, Y = np.meshgrid(X, Y)
Z = hz93.graph_mueff_ctr(X,Y,v,r)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sur = ax.plot_surface(X,Y,Z)
plt.show()

# 2D graph
# X = np.arange(0.01, 10, 0.01)
# z = some function
# plt.plot(X,z)
# plt.show()