import numpy as np

# ---------- #calculating (k and mu) from (e and v) ------------------
# :: inputs for both functions ([e1,e2,e3,...], [v1,v2,v3,...]). outputs [k1,k2,k3,...] and [mu1,mu2,mu3,...] respectively ::

def k(e,v):
    k = []
    for i in range(len(e)):
        k.append(e[i]/(3*(1-2*v[i])))
    return k

def mu(e,v):
    mu = []
    for i in range(len(e)):
        mu.append(e[i]/(2*(1-v[i])))
    return mu

# --------calculating keff from i(number of phases - 1), k, mu and r(radius of each phase)---------------------

# calculates the matrix N (equation 12)
# ::inputs (i "the corresponding phase - 1", [k1,k2,...], [mu1,mu2,...], [r1, r2, ...]) and outputs the matrix N(i+1)

def n(i,k,mu,r):
    ni = np.array([[3*k[i]+4*mu[i+1],(4/r[i]**3)*(mu[i+1]-mu[i])],[3*(k[i+1]-k[i])*r[i]**3,3*k[i+1]+4*mu[i]]])
    return ni * (1/(3*k[i+1]+4*mu[i+1]))

# calculates the matrix Q (equation 13)
# ::inputs (l "the number of phases - 1", [k1,k2,...], [mu1,mu2,...], [r1, r2, ...]) and outputs the matrix Q(l+1)

def q(l,k,mu,r):
    qi = np.array([[1,0],[0,1]])
    for i in range(l):
        qi = np.matmul(n(i,k,mu,r),qi)
    return qi

# calculates the effective propriety Keff (equation 45)
# ::inputs (l "the number of phases - 1", [k1,k2,...], [mu1,mu2,...], [r1, r2, ...]) and outputs the effective propriety Keff

def keff(l,k,mu,r):
    qi = q(l,k,mu,r)
    ke = (3*k[l]*r[l]**3*qi[0][0]-4*mu[l]*qi[1][0]) / (3*(r[l]**3*qi[0][0]+qi[1][0]))
    return ke

# calculates the effective propriety Keff (by homogenizing only 2 phases at a time)
# ::inputs (l "the number of phases - 1", [k1,k2,...], [mu1,mu2,...], [r1, r2, ...]) and outputs the effective propriety Keff

def keff_recu(l,k,mu,r):
    if l == 1:
        return keff(1,k,mu,r)
    else:
        return keff(1, [keff_ouss(l-1,k,mu,r),k[l]], [mu[l-1],mu[l]], [r[l-1],r[l]])

# calculates the effective propriety Keff (by homogenizing only 2 phases at a time) (equation 46)
# ::inputs (l "the number of phases - 1", [k1,k2,...], [mu1,mu2,...], [r1, r2, ...]) and outputs the effective propriety Keff

def keff_recu_simp(l,k,mu,r):
    if l == 0:
        return k[l]
    else:
        ke = k[l] + ((r[l-1]**3 / r[l]**3) / ((1/(keff_recu(l-1,k,mu,r)-k[l])) + ((3*(r[l]**3 - r[l-1]**3))/(r[l]**3 * (3*k[l] + 4*mu[l]))) ))
        return ke

# calculates the effective propriety Keff only for the case of 3 phases (by homogenizing only 2 phases at a time) (equation 47)
# ::inputs ([k1,k2,...], [mu1,mu2,...], [r1, r2, ...]) and outputs the effective propriety Keff

def keff3(k,m,r):
    a = ((3*k[2]+4*m[2])*r[1]**3)*((k[0]-k[1])*r[0]**3*(3*k[2]+4*m[1]) + (k[1]-k[2])*r[1]**3*(3*k[0]+4*m[1]))
    b = 3*(k[1]-k[0])*r[0]**3*(r[1]**3*(3*k[2]+4*m[1]) + 4*r[2]**3*(m[2]-m[1])) + (3*k[0]+4*m[1])*r[1]**3*(3*r[1]**3*(k[2]-k[1])+r[2]**3*(3*k[1]+4*m[2]))
    return k[2] + a/b

# --------calculating mueff from n(number of phases - 1), m (mu), v and r(radius of each phase)---------------------

# calculates the matrix L (equation 24)
# ::inputs (n "the corresponding phase - 1", [k1,k2,...], [v1,v2,...], [r1, r2, ...]) and outputs the matrix L(n+1)
def l(n,m,v,r):
    l1 = [r, -((6 * v[n] * r ** 3) / (1 - 2 * v[n])), 3 / r ** 4, (5 - 4 * v[n]) / ((1 - 2 * v[n]) * r ** 2)]
    l2 = [r, -((7 - 4 * v[n]) * r ** 3) / (1 - 2 * v[n]), -2 / r ** 4, 2 / r ** 2]
    l3 = [m[n], ((3 * v[n] * r ** 2 * m[n]) / (1 - 2 * v[n])), -(12 * m[n]) / r ** 5,
          ((2 * (v[n] - 5)) * (m[n] / r ** 3)) / (1 - 2 * v[n])]
    l4 = [m[n], -((7 + 2 * v[n]) * (r ** 2 * m[n])) / (1 - 2 * v[n]), (8 * m[n]) / r ** 5,
          (2 * (1 + v[n]) * (m[n] / r ** 3)) / (1 - 2 * v[n])]
    li = np.array([l1, l2, l3, l4])
    return li

# calculates the matrix M (equation 26)
# ::inputs (n "the corresponding phase - 1", [mu1,mu2,...], [v1,v2,...], [r1, r2, ...]) and outputs the matrix M(n+1)

def mk(n,m,v,r):
    mi = np.matmul(np.linalg.inv(l(n+1,m,v,r[n])),l(n,m,v,r[n]))
    return mi

# calculates the matrix P (equation 29)
# ::inputs (n "the number of phases - 1", [mu1,mu2,...], [v1,v2,...], [r1, r2, ...]) and outputs the matrix P(n+1)
def p(n,m,v,r):
    pi = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    for i in range(n):
        pi = np.matmul(mk(i,m,v,r),pi)
    return pi

# calculates the coefficient Z(alpha,beta) (equation 29)
# ::inputs (P (the matrix P(number of phases - 1) from the equation 29, a (the coefficient alpha), B (the coefficient beta)) and outputs coefficient z(alpha,beta)
def z(p,a,b):
    zi = p[a-1][0]*p[b-1][1] - p[b-1][0]*p[a-1][1]
    return zi

# calculates the coefficients A, B, C (equation 51)
# ::inputs (n "the number of phases - 1", [mu1,mu2,...], [v1,v2,...], [r1, r2, ...]) and outputs a list of the coefficient [A, B, C]
def coff(n,m,v,r):
    pi = p(n,m,v,r)
    a = 4*r[n]**10*(1-2*v[n])*(7-10*v[n])*z(pi,1,2) + 20*r[n]**7*(7-12*v[n]+8*v[n]**2)*z(pi,4,2) + 12*r[n]**5*(1-2*v[n])*(z(pi,1,4)-7*z(pi,2,3)) + 20*r[n]**3*(1-2*v[n])**2*z(pi,1,3) + 16*(4-5*v[n])*(1-2*v[n])*z(pi,4,3)
    b = 3*r[n]**10*(1-2*v[n])*(15*v[n]-7)*z(pi,1,2) + 60*r[n]**7*(v[n]-3)*v[n]*z(pi,4,2) - 24*r[n]**5*(1-2*v[n])*(z(pi,1,4)-7*z(pi,2,3)) - 40*r[n]**3*(1-2*v[n])**2*z(pi,1,3) - 8*(1-5*v[n])*(1-2*v[n])*z(pi,4,3)
    c = -r[n]**10*(1-2*v[n])*(7+5*v[n])*z(pi,1,2) + 10*r[n]**7*(7-v[n]**2)*z(pi,4,2) + 12*r[n]**5*(1-2*v[n])*(z(pi,1,4)-7*z(pi,2,3)) + 20*r[n]**3*(1-2*v[n])**2*z(pi,1,3) - 8*(7-5*v[n])*(1-2*v[n])*z(pi,4,3)
    return [a,b,c]

# calculates the effective propriety mueff (the valid solution from equation 51)
# ::inputs (n "the number of phases - 1", [mu1,mu2,...], [v1,v2,...], [r1, r2, ...]) and outputs the effective propriety mueff
def mueff(n,m,v,r):
    s = np.roots(coff(n,m,v,r))
    return max(s) * m[n]

# calculates the effective propriety Mueff (by homogenizing only 2 phases at a time)
# ::inputs (l "the number of phases - 1", [mu1,mu2,...], [v1,v2,...], [r1, r2, ...]) and outputs the effective propriety mueff
def mueff_recu(l,mu,v,r):
    if l == 1:
        return mueff(1,mu,v,r)
    else:
        return mueff(1, [mueff_ouss(l-1,mu,v,r),mu[l]], [v[l-1],v[l]], [r[l-1],r[l]])

# calculates the effective propriety Mueff (by homogenizing only 2 phases at a time) iteratively
# ::inputs (l "the number of phases - 1", [mu1,mu2,...], [v1,v2,...], [r1, r2, ...]) and outputs the effective propriety mueff
def mueff_recu_simp(l,mu,v,r,k):
    mf = mueff(1,mu,v,r)
    kf = keff(1,k,mu,r)
    for i in range(l-1):
        vf = ((mf/kf)-(3/2))/(((mf/kf)-(3/2))-1)
        mf = mueff(1,[mf,mu[i+2]],[vf,v[i+2]],[r[i+1],r[i+2]])
        kf = keff(1, [kf,k[i+2]], [mu[i+1],mu[i+2]], [r[i+1],r[i+2]])
    return mf

# functions for 3D plotting

# to show the volume fraction effect, here 3 phase model is considered
# k and mu are fixed, x and y axis represents the volume fraction of phase 1 and 2 respectively, z represents the effective propriety k
def graph_keff_fv(X,Y,k,mu):
    z = []
    for i in range(len(X)):
        z.append([])
        for j in range(len(X[i])):
            if (X[i][j] + Y[i][j]) >= 1:
                z[-1].append(np.nan)
            else:
                z[-1].append(keff(2,k,mu,[((3/(4*np.pi))*X[i][j])**(1/3),((3/(4*np.pi))*(X[i][j]+Y[i][j]))**(1/3),(3/(4*np.pi))**(1/3)]))
    return np.array(z)

# to show the volume fraction effect, here 3 phase model is considered
# k and mu are fixed, x and y axis represents the volume fraction of phase 1 and 2 respectively, z represents the effective propriety mu
def graph_mueff_fv(X,Y,mu,v):
    z = []
    for i in range(len(X)):
        z.append([])
        for j in range(len(X[i])):
            if (X[i][j] + Y[i][j]) >= 1:
                z[-1].append(np.nan)
            else:
                z[-1].append(mueff(2,mu,v,[((3/(4*np.pi))*X[i][j])**(1/3),((3/(4*np.pi))*(X[i][j]+Y[i][j]))**(1/3),(3/(4*np.pi))**(1/3)]))
    return np.array(z)

# to show the contrast effect, here 2 phase model is considered
# r and v are fixed, x and y axis represents the propriety E of phase 1 and 2 respectively, z represents the effective propriety k
def graph_keff_ctr(X,Y,v,r):
    z = []
    for i in range(len(X)):
        z.append([])
        for j in range(len(X[i])):
            z[-1].append(keff(1,k([X[i][j],Y[i][j]],v),mu([X[i][j],Y[i][j]],v),r))
    return np.array(z)

# to show the contrast effect, here 2 phase model is considered
# r and v are fixed, x and y axis represents the propriety mu of phase 1 and 2 respectively, z represents the effective propriety mu
def graph_mueff_ctr(X,Y,v,r):
    z = []
    for i in range(len(X)):
        z.append([])
        for j in range(len(X[i])):
            z[-1].append(mueff(1,mu([X[i][j],Y[i][j]],v),v,r))
    return np.array(z)

