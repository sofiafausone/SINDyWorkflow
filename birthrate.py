
# In[1]:
import numpy as np
from scipy.integrate import odeint, quad
import math 
import matplotlib.pyplot as plt
import csv
import pysindy as ps
import pandas as pd 
from numpy import diff
import math
import scipy.integrate as int
import matplotlib
matplotlib.rcParams.update({'font.size': 13})

# In[2]:
#coefficients for unbounded state, where v2k1 < v1k2

a = 1
b = 1/100
N0=10

#creating analytic solution data
def sol(t):
    N = a/b + (N0-a/b)*np.exp(-b*t)
    return N


print('correct result')
print('dN/dt= ',a, '-',b, 'N')



datapoints = 1001
nsol = np.empty(datapoints)
t = np.linspace(0, 1000, datapoints)

for i in range(len(t)): 
        nsol[i] = sol(t[i])

print(t)
print(nsol)
print(len(t), len(nsol))

plt.plot(t, nsol)

# In[2]:
#getting 
file_path = '/Users/sofiafausone/HowardLab/ATSBL/bdData/Transient/a1_b0.01_n10N1000_T2.csv'
file_path2 = '/Users/sofiafausone/HowardLab/ATSBL/bdData/Transient/Time Series/TIME_a1_b0.01_n10N1000_T2.csv'
#file_path = '/Users/sofiafausone/HowardLab/pysindy/examples/2_introduction_to_sindy/birthdeath/bdNormal.csv'
#file_path2 = '/Users/sofiafausone/HowardLab/pysindy/examples/2_introduction_to_sindy/birthdeath/bdtimeNormal.csv'
# Read the CSV file into a pandas DataFrame

data_frame = pd.read_csv(file_path)
time_df = pd.read_csv(file_path2, header = None)
N = data_frame.to_numpy()
time = time_df.to_numpy()[0][0:-1]

print(N.shape)
print(time.shape)

# In[2]:

N1 =N[:,0]
Navg = np.mean(N, axis =1)
plt.plot(time, N1, label = 'Simulation Data Run 1')
plt.plot(time, Navg, label = 'Simulation Data Average, 1000 runs', linewidth = 2.5)

plt.plot(t, nsol, label = 'Analytic Solution')
plt.title('Birth-Death Model')
plt.xlabel('Time (s)')
plt.plot((450, 450), (85, 115), scaley = False, color = 'k')
plt.ylabel('Population Number')
plt.text(0, 90, 'Transient', fontsize=14,
        verticalalignment='top')
plt.text(550, 90, 'Steady State', fontsize=14,
        verticalalignment='top')
plt.legend()


# In[2]:

#transient data 
transient = 400
Navg_t = Navg[0:transient*100]
time_t = time[0:transient*100]
nsol_t = nsol[0:transient]
t_t = t[0:transient]
plt.plot (time_t, Navg_t, label = 'Simulation Data Average, Transient', linewidth = 2.5)
plt.plot(t_t, nsol_t, label = 'Analytic Solution, Transient', linewidth = 1.5)
plt.legend()

#steady state data
ss = 400
Navg_s = Navg[ss*100:]
time_s = time[ss*100:]
nsol_s = nsol[ss:]
t_s = t[ss:]
plt.plot (time_s, Navg_s, label = 'Simulation Data Average, SS', color = 'm')
plt.plot(t_s, nsol_s, label = 'Analytic Solution, SS', color = 'g')
plt.legend()


# %%
#averages over different N

N10 = N[:,0:10]
N100 = N[:,0:100]
N1000 = N

N10avg = np.mean(N10, axis =1)
N100avg = np.mean(N100, axis =1)
N1000avg = np.mean(N, axis =1)

plt.plot(time, N1, label = 'Simulation Data [1 Run]')
plt.plot(time, N10avg, label = 'Sim [10 Run Avg]')
plt.plot(time, N100avg, label = 'Sim [100 Run Avg]')
plt.plot(time, N1000avg, label = 'Sim [1000 Run Avg]', color = 'm')

plt.plot(t, nsol, linestyle = 'dashed', color = 'k', label = 'Analytic Solution')
plt.legend()
plt.title('Birth-Death Model: dN/dt = 1-0.1N')
plt.xlabel('Time (s)')
plt.ylabel('Change in Population Number')
plt.plot((450, 450), (85, 115), scaley = False, color = 'k')

plt.text(0, 90, 'Transient', fontsize=14,
        verticalalignment='top')
plt.text(550, 90, 'Steady State', fontsize=14,
        verticalalignment='top')
# %%

#test pysindy on the analytic data
u = N1

differentiation_method = ps.FiniteDifference(order=2)
feature_library = ps.PolynomialLibrary(degree=5)
optimizer = ps.STLSQ(threshold=0.005)

model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
    feature_names=["t"],
)


model.fit(u, t= time)
model.print()


# %%
#test transient vs steady state

u = Navg_s
differentiation_method = ps.FiniteDifference(order=2)
feature_library = ps.PolynomialLibrary(degree=5)
optimizer = ps.STLSQ(threshold=0.005)

model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
    feature_names=["t"],
)


model.fit(u, t= time_s)
model.print()



# %%

SSE = [0.213, 0.00865, 0.000289, 0.000169]
nums = [1,10,100,1000]
plt.semilogx(nums, SSE, marker = 'o')
plt.title('Sindy Model Error for Different Total Runs Averaged')
plt.xlabel('Runs')
plt.ylabel('Sum of Squares Error')



# In[2]:
#test pysindy on simulated data 

u = np.transpose(Navg_s)
fd = ps.FiniteDifference()
diff = fd._differentiate(u, time)
#diff = ps.FiniteDifference(u,time, order=1)
print(diff)
differentiation_method = ps.FiniteDifference(order=1)
feature_library = ps.PolynomialLibrary(degree=1)
optimizer = ps.STLSQ(threshold=0.0)

model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
    feature_names=["t"],
)


model.fit(u, t= t_s)
model.print()

#for i in range(len(x)): 
 #       pplus[i] = G(x[i])
  #      pminus[i] = S(x[i])


# In[2]:

functions = [lambda x: x, lambda x : np.exp(-x)]
#functions = [lambda x : x, lambda x : np.gradient(x)/(np.gradient(len) + 0.000001)]
lib = ps.CustomLibrary(library_functions=functions)


optimizer = ps.STLSQ(threshold=0.16)
differentiation_method = ps.FiniteDifference(order=2)
model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=lib,
    optimizer=optimizer,
    feature_names=["pp", "pm"],
)
model.fit(u)
model.print()

# In[2]:




library_functions = [
    lambda x: x
]
library_function_names = [
    lambda x: x
]

pde_lib = ps.PDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    derivative_order=1,
    spatial_grid=x,
    include_bias = True,
    include_interaction = False,
    is_uniform=True,
    periodic=False
)
#
#feature_library = ps.GeneralizedLibrary([pde_lib,lib])
feature_library = lib
print('STLSQ model: ')
optimizer = ps.STLSQ(threshold=0.0, alpha=1e-5, 
                     normalize_columns=True, max_iter=2000, verbose = False)
model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)
model.fit(u)
#model.fit(u, t = t)
model.print()


# In[2]:
print(time)

#X = np.stack((pplus,pminus), axis=-1)  # First column is t1, second is t2
#differentiation_method = ps.FiniteDifference(order=2)
# We can select a differentiation method from the `differentiation` submodule.


# In[2]:
#u = pplus
u = np.column_stack((pplus, pminus))
print(u.shape)
print(time)
#u = u.reshape(len(lens), len(time), 1)

dt = time[1] - time[0]


#library_functions = [lambda x: x, lambda x: x * x]
#library_function_names = [lambda x: x, lambda x: x + x]
library_functions = [lambda x: x]
library_function_names = [lambda x: x]
pde_lib = ps.PDELibrary(library_functions=library_functions, 
                        function_names=library_function_names, 
                        derivative_order=1, spatial_grid=lens, 
                        include_bias=True, is_uniform=True)
poly_library = ps.PolynomialLibrary(degree=2)
lib = ps.GeneralizedLibrary([poly_library, pde_lib])


# Fit the model with different optimizers.
# Using normalize_columns = True to improve performance.
print('STLSQ model: ')
optimizer = ps.STLSQ(threshold=0.1, alpha=1e-5, normalize_columns=True)
model = ps.SINDy(feature_library=lib, optimizer=optimizer)
model.fit(u, t=dt)
model.print()





# In[2]:

#u = ps.AxesArray(np.ones((len(lens) * len(time), 2)),{"ax_coord":1})
library_functions = [lambda x: x]
library_function_names = [lambda x: x]
pde_lib = ps.PDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    derivative_order=1,
    spatial_grid=lens,
).fit(u)
print("1st order derivative library with function names: ")
print(pde_lib.get_feature_names(), "\n")


# %%
dt = time[1] - time[0]
print(dt)
#u = u.reshape(len(lens), len(time), 1)
optimizer = ps.STLSQ(threshold=0.2, alpha=1e-5, normalize_columns=True)
#optimizer = ps.STLSQ(threshold=0.2)
model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
model.fit(u, t=dt)
model.print()



# %%

print(time.shape)

# %%

u = np.zeros((36000, 36000, 2))
print('test1')
u[:, :, 0] = pplus
print('test2')
u[:, :, 1] = pminus
print('test3')


# %%
dt = 0.1
#differentiation_method = ps.FiniteDifference(order=2)
#u = np.column_stack((pplus, pminus))
#u_dot = ps.FiniteDifference(axis=2)._differentiate(u, dt)
# Odd polynomial terms in (u, v), up to second order derivatives in (u, v)
library_functions = [
    lambda x: x,
    lambda x: x * x,
    lambda x, y: x * y,
    lambda x, y: y * y,
]
library_function_names = [
    lambda x: x,
    lambda x: x + x + x,
    lambda x, y: x + y + y,
    lambda x, y: x + x + y,
]
pde_lib = ps.PDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    derivative_order=2,
    spatial_grid=lens,
    include_bias=True,
    is_uniform=True,
    periodic=True
)
print('STLSQ model: ')
optimizer = ps.STLSQ(threshold=0.6, alpha=1e-5, 
                     normalize_columns=True, max_iter=200)
model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
model.fit(u, t =dt)
model.print()
# %%
