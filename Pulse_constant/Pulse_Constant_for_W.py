
# librerias
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix,negativity,state_fidelity,Operator, concurrence
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

def esque_w(params):
    """Primer bloque
       t_0 : duración de los canales        d0,d1
       t_1 : duración del canal             u0
       A0  : Amplitude del primer pulso     d0
       A1  : Apmpliud del segundo pulso     d1
       A2  : Amplitud del pulsos de control U0
       Segundo bloque
       t_2 : duracion del canal             d2
       t_3 : duracion del mcanal            u1
       A3  : Amplitud del pulso             d2
       A4  : Amplitud del pulso             u1
       Tercer bloque
       t_4 : duración del canal             d0
       t_5 : duracion del canal             d1
       t_6 : duracion del canal             u0
       A5  : Amplitud del canal             d0
       A6  : Amplitud del canal             d1
       A7  : Amplitud del canal             u0 
       """
    t_0, t_1, t_2,t_3, t_4, t_5 ,A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11  = params


    # Convertir a enteros por requerimiento del backend
    t_0 = int(t_0)
    t_1 = int(t_1)
    t_2 = int(t_2)
    t_3 = int(t_3)
    t_4 = int(t_4)
    t_5 = int(t_5)
    
    

    with pulse.build(name="pulso w") as kp:
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)
        u0 = pulse.ControlChannel(0)
        u1 = pulse.ControlChannel(1)
        
        # pulse.shift_phase(p0,d0)
        pulse.play(pulse.Constant(t_0,A0),d0)
        pulse.play(pulse.Constant(t_0,A1),d1)
        pulse.play(pulse.Constant(t_0,A2),d2)
        
        pulse.delay(duration = 1 + t_0, channel=u0)
        pulse.play(pulse.Constant(duration = t_1 , amp=A3), u0)
        
        pulse.delay( t_1 + 1, d0)
        pulse.play(pulse.Constant(t_2,A4),d0)
        
        pulse.delay(t_1 +1, d1)
        pulse.play(pulse.Constant(t_2,A5),d1)
        
        pulse.delay(t_1 +1, d2)
        pulse.play(pulse.Constant(t_2,A6),d2)
        
        
        pulse.delay(t_0 + t_1 + 1 + 1 + t_2, u1)
        pulse.play(pulse.Constant(t_3,A7),u1)
        
        pulse.delay(t_3+1,d0)
        pulse.play(pulse.Constant(t_4,A8),d0)
        
        pulse.delay(t_3+1,d1)
        pulse.play(pulse.Constant(t_4,A9),d1)
        
        pulse.delay(t_3+1,d2)
        pulse.play(pulse.Constant(t_4,A10),d2)
        
        pulse.delay(1+t_2+1+t_3+t_4 +1,u0)
        pulse.play(pulse.Constant(t_5,A11),u0)
        
        
    return kp

from scipy.optimize import differential_evolution


bounds = [(230,300),       # t_0
          (1500,1800),      # t_1
          (180,200),       # t_2
          (1200,2000),      # t_3
          (270,300),       # t_4
          (1300,2000),      # t_5
          (-0.9,0.9),      # A0
          (-0.9,0.92772),  # A1
          (-0.9726,0.962552), # A2
          (-0.9,0.92322),     # A3
          (-0.9,0.92322),     # A4
          (-0.9,0.92322),     # A5
          (-0.9,0.92322),     # A6
          (-0.9,0.92322),     # A7
          (-0.9,0.92322),     # A8
          (-0.9,0.92322),      # A9
         (-0.9,0.92322),      # A10
          (-0.9,0.92322)      # A11
          ]






iteracion = {"n":0} 




def objective_w(x):
    signal = esque_w(x)
    tiempo_t = signal.duration  # duración total del pulso en pasos de dt
    sol = solver.solve(
        t_span=[0, tiempo_t*dt],
        y0=y0,
        signals=signal,
        method="jax_odeint",
        atol=1e-8,
        rtol=1e-9
    )
    yf = sol.y[-1]/np.linalg.norm(sol.y[-1])
    
    conc_01 = concurrence(partial_trace(yf, [2]))
    conc_02 = concurrence(partial_trace(yf, [1]))  
    conc_12 = concurrence(partial_trace(yf, [0]))
    conc_sum = conc_01 + conc_02 + conc_12
    
    V = ((conc_01 - 2/3)**2 + (conc_02 - 2/3)**2 + (conc_12 - 2/3)**2)/3
    
    total_cost = - conc_sum + 10*V
    
    cost_func = total_cost
    return cost_func


def callback(xk, convergence):
    iteracion["n"] += 1
    costo = objective_w(xk)
    print(f"{iteracion['n']:3d} | Cost: {costo:.6f} | x: {xk}")
    
    
result = differential_evolution(
    objective_w,
    bounds=bounds,
    workers=1,          # ✅ ejecuta en serie: seguro con JAX
    strategy='currenttobest1bin',
    maxiter=15,
    tol=1e-7,
    mutation=(0.5, 1.0),
    recombination=0.7,
    polish=True,
    disp=True,
    callback=callback
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
