import numpy as np 
from qiskit.quantum_info import Operator, Statevector, negativity, DensityMatrix, partial_trace
from qiskit_dynamics import Solver, Signal
from scipy.optimize import minimize
import sympy as sym

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit_dynamics.models import HamiltonianModel

from qiskit_ibm_runtime.fake_provider import  FakeSherbrooke
from qiskit_ibm_runtime import QiskitRuntimeService
import qiskit.pulse as pulse

import matplotlib.pyplot as plt

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Desactiva GPU]
import jax
jax.config.update('jax_platform_name', 'cpu')  # Fuerza CPU
# 
#  Llamar backend con la cuenta

backend = FakeSherbrooke()

#configuracion de backend
propiedades = backend.properties()
configuracion = backend.configuration()


Hz = 1e9
# Definicion de la frecuencia de los qubits en GHz
w7   = configuracion.hamiltonian.get("vars").get("wq7")
w8   = configuracion.hamiltonian.get("vars").get("wq8")
w9   = configuracion.hamiltonian.get("vars").get("wq9")
# anamornicidad de los qubits en GHz
delta7 = configuracion.hamiltonian.get("vars").get("delta7")
delta8 = configuracion.hamiltonian.get("vars").get("delta8")
delta9 = configuracion.hamiltonian.get("vars").get("delta9")
# acoples en GHz
J78 = configuracion.hamiltonian.get("vars").get("jq7q8")
J89 = configuracion.hamiltonian.get("vars").get("jq8q9")
# fuerza de control en GHz
r0   =  configuracion.hamiltonian.get("vars").get("omegad7")
r1   =  configuracion.hamiltonian.get("vars").get("omegad8")
r2   =  configuracion.hamiltonian.get("vars").get("omegad9")
# tiempo de gate en ns
dt = backend.dt*1e9


v7 = w7 /(2*np.pi)
v8 = w8 /(2*np.pi)
v9 = w9 /(2*np.pi)


# Definir los operadores 
dim = 3
a = np.diag(np.sqrt(np.arange(1, dim)), 1)
adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
N = np.diag(np.arange(dim))

ident = np.eye(dim, dtype=complex)
full_ident = np.eye(dim**2, dtype=complex)

N0 = np.kron(ident,np.kron(ident, N))
N1 = np.kron(ident,np.kron(N, ident))
N2 = np.kron(np.kron(N, ident),ident)

a0 = np.kron(ident,np.kron(ident, a))
a1 = np.kron(ident,np.kron(a, ident))
a2 = np.kron(np.kron(a, ident),ident)


a0dag = np.kron(ident,np.kron(ident, adag))
a1dag = np.kron(ident,np.kron(adag, ident))
a2dag = np.kron(np.kron(adag, ident),ident)


# Haniltoniano del sistema
static_ham = 2*np.pi*v7 * N0+   delta7*(a0dag@a0dag@a0@a0)/2   + 2*np.pi*v8 * N1+   delta8*(a1dag@a1dag@a1@a1)/2   + 2*np.pi*v9* N2 + delta9*(a2dag@a2dag@a2@a2)/2  + J78*(a0dag @ a1 +  a0 @ a1dag) + J89*(a1dag @ a2 +  a1 @ a2dag)

# Hamiltonianos de controles
H_drive7 =  r0 * (a0 + a0dag)
H_drive8 =  r1 * (a1 + a1dag)
H_drive9 =  r2 * (a2 + a2dag)
  

y0 = Statevector([1,0,0]).tensor(Statevector([1,0,0])).tensor(Statevector([1,0,0]))
# y0  = Statevector.from_label("000")

  


# Solucionar la ecuación de Schrödinger
solver  = Solver(static_hamiltonian=static_ham,
                 hamiltonian_operators=[H_drive7, H_drive8, H_drive9,H_drive7,H_drive8],
                 hamiltonian_channels=["d0", "d1", "d2","u0","u1"],
                 channel_carrier_freqs= {"d0": v7, "d1": v8, "d2": v9, "u0": v8, "u1": v9},
                 dt=dt,
                 array_library="jax")



# pulsos
from qiskit import pulse

def esque_pul(params):
    """Create a pulse schedule for a Bell state.

    Args:
        params (list): A list containing the parameters for the Gaussian pulses.

    Returns:
        _type_: Sigma1, sigma2, duration1, duration2,sigma3, duration3,
        
        Corresponding to each Gaussian pulse respectively.
        
        dura1, sigma1: Parameters for the first Gaussian pulse
        
        dura2, sigma2: Parameters for the second Gaussian pulse
        
        dura3, sigma3: Parameters for the third Gaussian pulse.
        
        
    1st pulse on d0 and d1
    
    2nd pulse on u0
    
    dura1 > with1
    
    dura2 > with1
    
    dura3 > with3
    
    1st and 2nd pulse with the same width
    
    in this case with1 = 191.910934 and with3 = 797.242687
    
    Amplitudes are fixed based on previous optimizations.
    """
    dura1, sigma1, dura3, sigma3,  = params
    
    dura1 = int(dura1)
    dura2 = int(dura1)
    dura3 = int(dura3)
    sigma1 = int(sigma1)
    sigma2 = int(sigma1)
    sigma3 = int(sigma3)
    
    with1 =  191.910934
    with3 =  797.242687
    
    with1 = int(with1)
    with3 = int(with3)
    
    amp1  =  0.6824432988764655
    amp2  =  0.8286644926788276
    amp3  =  0.7868326972398657
    
    dura = int(dura2/2)
   
    
    with pulse.build(name="pulse Bell") as bell:
        
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        u0 = pulse.ControlChannel(0)
        

        # primer pulso en d0 y d1
        pulse.play(pulse.GaussianSquare(dura1, amp1, sigma1, with1), d0)
        pulse.play(pulse.GaussianSquare(dura2, amp2, sigma2, with1), d1)
        
        # pulso de control en u0
        pulse.delay(with1+  dura, u0)
        pulse.play(pulse.GaussianSquare(dura3, amp3, sigma3, with3), u0)
    
    return bell
   

from scipy.optimize import differential_evolution


bounds = [(200, 400), 
          (40, 100), 
          (800, 1200), 
          (40, 100)]

iteracion = {"n":0} 

# fun_objetivo
def evo_bell_qubit(x):
    signal = esque_pul(x)
    tiempo_t = signal.duration  # duración total del pulso en pasos de dt
    sol = solver.solve(
        t_span=[0, tiempo_t*dt],
        y0=y0,
        signals=signal,
        method="jax_odeint",
        atol=1e-7,
        rtol=1e-8
    )
    yf = sol.y[-1]/np.linalg.norm(sol.y[-1])


    redu_vector = [] 
    for i in [0,1,3,4,9,10,12,13]:
        redu_vector.append(yf[i])
    yr = np.array(redu_vector)/np.linalg.norm(redu_vector)



    # rho_bc = partial_trace(Statevector(yr), [0])
    # rho_ac = partial_trace(Statevector(yr), [1])
    rho_ab = partial_trace(Statevector(yr), [2])

    cost = (negativity(rho_ab,[0]) -  0.5)**2 + (negativity(rho_ab,[1]) -  0.5)**2 
    return cost

def callback(xk, convergence):
    iteracion["n"] += 1
    costo = evo_bell_qubit(xk)
    print(f"{iteracion['n']:3d} | Cost: {costo:.6f} | x: {xk}")
    
    
result = differential_evolution(
    func= evo_bell_qubit,
    bounds= bounds,
    callback=callback,
    disp=True,
    seed= 20,
    maxiter= 50
)    


# Resultado final
print("\nResultado final:")
print("x óptimo:", result.x)
print("Costo mínimo:", result.fun)

import json
import os

# Crear diccionario con los resultados
resultados = {
    "parametros_optimos": result.x.tolist(),
    "costo_minimo": result.fun,
    "iteraciones": iteracion["n"],
    "success": result.success,
    "message": result.message
}

# Crear carpeta si no existes
carpeta_salida = "resultados_optimizacion"
os.makedirs(carpeta_salida, exist_ok=True)

# Guardar archivo
nombre_archivo = os.path.join(carpeta_salida, "resultado_optimo.json")
with open(nombre_archivo, "w") as f:
    json.dump(resultados, f, indent=4)

print(f"\n✅ Resultados guardados en '{nombre_archivo}'")


