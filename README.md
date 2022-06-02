# Herve-and-Zaoui-Model
A python implementation of the Herve and Zaoui analytical model for calculating homogenized elastic proprieties of composite material with spherical n-layered inclusion.

You can find the model's paper at: https://www.sciencedirect.com/science/article/abs/pii/0020722593900594

Almost every function in the code is an implementation of an equations in the paper, with the relevant eqution's number is found in the commecnt right above the function.

Since the model makes use of the bulk, shear modulus and the poisson's ratio, functions for calculating the bulk and shear modulus from young's modulus and the poisson's ratio in the isotropic case were added at the top of the code.

At the bottom of the script you can find some helpful functions for 3D plotting.

In the code the following terminology is considered:

k: Bulk modulus.

mu (sometimes just m): shear modulus.

v: young's modulus.

r: the radius.

n,l,i : number of the phases or the corresponding phase accordingly.

# For questions and/or collaboration please contact me at oussamamedounissi@gmail.com.
