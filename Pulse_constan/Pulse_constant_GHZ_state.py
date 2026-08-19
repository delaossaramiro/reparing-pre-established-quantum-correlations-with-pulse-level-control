# librerias
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix,negativity,state_fidelity,Operator
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


from qiskit_dynamics import Solver, Signal
from qiskit_dynamics.backend import parse_backend_hamiltonian_dict

from qiskit import pulse
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import random
import sympy as sym

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Llamar fake backend
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
dim = 2
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
static_ham = 2*np.pi*v7 * N0  +2*np.pi*v8 * N1  + 2*np.pi*v9* N2   + J78*(a0dag @ a1 +  a0 @ a1dag) + J89*(a1dag @ a2 +  a1 @ a2dag)

# Hamiltonianos de controles
H_drive7 =  r0 * (a0 + a0dag)
H_drive8 =  r1 * (a1 + a1dag)
H_drive9 =  r2 * (a2 + a2dag)
  

y0  = Statevector.from_label("000")
# ==== Solver ====
solver = Solver(
    static_hamiltonian=static_ham,
    hamiltonian_operators=[H_drive7, H_drive8, H_drive9, H_drive7, H_drive8],
    hamiltonian_channels=["d0", "d1", "d2", "u0", "u1"],
    channel_carrier_freqs= {"d0": v7, "d1": v8, "d2": v9, "u0": v8, "u1": v9},
    dt=dt,
    array_library="jax"
)



from qiskit import pulse

def esque_pul(params):
    t_0, t_1, t_2,t_3,A0, A1, A2, A3, A4, A5,  = params

    # Convertir a enteros por requerimiento del backend
    t_0 = int(t_0)
    t_1 = int(t_1)
    t_2 = int(t_2)
    t_3 = int(t_3)
    

    with pulse.build(name="pulso 3q") as kp:
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)
        u0 = pulse.ControlChannel(0)
        u1 = pulse.ControlChannel(1)

        # Primer pulso en d0 y d1
        pulse.play(pulse.Constant(duration=t_0, amp=A0), d0)
        pulse.play(pulse.Constant(duration=t_0, amp=A1), d1)
        
    
        # Delay en canal u0
        pulse.delay(duration=t_0 + 1, channel=u0)
        
        # Segundo pulso en d1 y u0
        pulse.play(pulse.Constant(duration=t_1 , amp=A2), u0)
        
       # delay pulso d2 y u1 
        pulse.delay(duration= (t_0+ t_1 + 1 ), channel= d2)
        pulse.delay(duration= (t_0+ t_1 + t_2 +1  ), channel= u1)
        
        
        # pulsos locales d1 y d2 
        pulse.play(pulse.Constant(duration=t_2, amp=A3),d2)
        # pulse.play(pulse.Constant(duration=t_2, amp=A4),d1)
        
       # pulso local d2
        pulse.play(pulse.Constant(duration=t_3, amp=A4),d2)
        
        
      # pulso u1
        pulse.play(pulse.Constant(duration=t_3, amp=A5),u1)
        
    return kp



from scipy.optimize import differential_evolution


bounds =   [(190,300),
          (200,2000),
          (150,300),
          (200,2000),
          (-0.9,0.9),
          (-0.9,0.92772),
          (-0.9726,0.962552),
          (-0.9,0.92322),
          (-0.9,0.92322),
          (-0.9,0.92322)]


iteracion = {"n":0} 


def objective_123(x):
    signal = esque_pul(x)
    tiempo_t = signal.duration  # duración total del pulso en pasos de dt
    sol = solver.solve(
        t_span=[0, tiempo_t*dt],
        y0=y0.data,
        signals=signal,
        method="jax_odeint",
        atol=1e-7,
        rtol=1e-8
    )
    
    t = sol.y[-1]/np.linalg.norm(sol.y[-1])
    
    
    d1 = (t[0] * t[7])**2 + (t[1] * t[6])**2 + (t[2] * t[5])**2 + (t[3] * t[4])**2
    d2 = (t[0] * t[7] * t[3] * t[4] + t[0] * t[7] * t[5] * t[2] +
                t[0] * t[7] * t[6] * t[1] + t[3] * t[4] * t[5] * t[2] +
                t[3] * t[4] * t[6] * t[1] + t[5] * t[2] * t[6] * t[1])
    d3 =  (t[0] * t[6] * t[5] * t[3]) + (t[7] * t[1] * t[2] * t[4])
    tabc = 4 * abs(d1 - 2 * d2 + 4 * d3)
    
    return 1-tabc 




def callback(xk, convergence):
    iteracion["n"] += 1
    costo = objective_123(xk)
    print(f"{iteracion['n']:3d} | Cost: {costo:.6f} | x: {xk}")
    
    
result = differential_evolution(
    func= objective_123,
    bounds= bounds,
    callback=callback,
    disp=True,
    seed=50,
    maxiter=30
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

# Crear carpeta si no existe
carpeta_salida = "resultados_optimizacion"
os.makedirs(carpeta_salida, exist_ok=True)

# Guardar archivo
nombre_archivo = os.path.join(carpeta_salida, "resultado_optimo.json")
with open(nombre_archivo, "w") as f:
    json.dump(resultados, f, indent=4)

print(f"\n✅ Resultados guardados en '{nombre_archivo}'")
