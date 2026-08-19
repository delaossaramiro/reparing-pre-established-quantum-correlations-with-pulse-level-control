# Quantum system
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace, Operator,DensityMatrix,state_fidelity,concurrence,entropy,negativity
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from qiskit_dynamics import Solver,Signal
import numpy as np
from scipy.optimize import rosen, differential_evolution

from qiskit import pulse
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize
import random
import sympy as sym 
import jax


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

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
print(v7, v8, v9)

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
# ==== Solver ====
# Solucionar la ecuación de Schrödinger
solver  = Solver(static_hamiltonian=static_ham,
                 hamiltonian_operators=[H_drive7, H_drive8, H_drive9,H_drive7,H_drive8],
                 hamiltonian_channels=["d0", "d1", "d2","u0","u1"],
                 channel_carrier_freqs= {"d0": v7, "d1": v8, "d2": v9, "u0": v8, "u1": v9},
                 dt=dt,
                 array_library="jax")

# === Pulsos ===
# Pulso GHZ

# Pulso GHZ

# Pulso GHZ

# pulso w  gaussian cuadrado

def esque_pul_w(params):
    # Pulso GHZ

    sigma1,amp1,amp2,amp3,  sigma4,amp4,amp5,amp6,  sigma7,amp7,amp8,amp9, sigma10,amp10,  sigma11,amp11, sigma12,amp12   = params
    # amp1, amp2,     amp3,     amp4 ,    amp5   = params
    
    # sigma1 = 1.106e+02
    # sigma3 = 6.505e+01
    # sigma4 = 4.095e+01
    # sigma5 = 2.973e+02

    sigma1 = int(sigma1)
    sigma2 = int(sigma1)
    sigma3 = int(sigma1)
    
    sigma4 = int(sigma4) 
    sigma5 = int(sigma4)
    sigma6 = int(sigma4)
    
    sigma7 = int(sigma7)
    sigma8 = int(sigma7)
    sigma9 = int(sigma7)
    
    sigma10 = int(sigma10)
    sigma11 = int(sigma11)
    sigma12 = int(sigma12)

    width1  =  2.34830783e+02
    width10 =  1.56574174e+03
    width4  =  1.87479735e+02
    width11 =  1.22942236e+03
    width7  =  2.74196599e+02
    width12 =  1.32352652e+03
    
  

    width1 = int(width1)
    width10 = int(width10)
    
    
    width4 = int(width4)
    width11 = int(width11)
    
    
    width7 = int(width7)
    width12 = int(width12)
    
    

    dura1 = width1 +  2*int(1*sigma1)
    dura2 = width1 +  2*int(1*sigma1)
    dura3 = width1 +  2*int(1*sigma1)
    
    dura10 = width10 +  2*int(1*sigma10)
    
    
    dura4 = width4 +  2*int(1*sigma4)
    dura5 = width4 +  2*int(1*sigma4)
    dura6 = width4 +  2*int(1*sigma4)
    
    dura11 = width11 +  2*int(1*sigma11)

    dura7 = width7 +  2*int(1*sigma7)
    dura8 = width7 +  2*int(1*sigma7)
    dura9 = width7 +  2*int(1*sigma7)
    
    dura12 = width12 +  2*int(1*sigma12)
    
    
      

    # amp1  =   4.90394435e-01
    # amp2  =  -6.91612520e-01
    # amp3  =   6.65492967e-01
    # amp4  =   1.65200193e-01
    # amp5  =   6.19735822e-01
    
    
    
    with pulse.build(name="w") as w:
        
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        u0 = pulse.ControlChannel(0)
        d2 = pulse.DriveChannel(2)
        u1 = pulse.ControlChannel(1)
        

        # primer pulso en d0 y d1
        pulse.play(pulse.GaussianSquare(dura1, amp1, sigma1, width1), d0)
        pulse.play(pulse.GaussianSquare(dura2, amp2, sigma2, width1), d1)
        pulse.play(pulse.GaussianSquare(dura3, amp3, sigma3, width1), d2)

        # pulso de control en u0
        pulse.delay( dura1 , u0)
        pulse.play(pulse.GaussianSquare(dura10, amp10, sigma10, width10), u0)

        # tercer pulso en d2
        pulse.delay(dura10  , d0)
        pulse.delay(dura10  , d1)
        pulse.delay(dura10  , d2)
        pulse.play(pulse.GaussianSquare(dura4, amp4, sigma4, width4), d0)
        pulse.play(pulse.GaussianSquare(dura5, amp5, sigma5, width4), d1)
        pulse.play(pulse.GaussianSquare(dura6, amp6, sigma6, width4), d2)
        # # # # cuarto pulso en u1
        
        pulse.delay(dura1 + dura10 + dura4, u1)
        pulse.play(pulse.GaussianSquare(dura11, amp11, sigma11, width11), u1)
        
        pulse.delay( dura11, d0)
        pulse.delay( dura11, d1)
        pulse.delay( dura11, d2)
        pulse.play(pulse.GaussianSquare(dura7, amp7, sigma7, width7), d0)
        pulse.play(pulse.GaussianSquare(dura8, amp8, sigma8, width7), d1)
        pulse.play(pulse.GaussianSquare(dura9, amp9, sigma9, width7), d2)

        pulse.delay(dura4 + dura11 +dura7, u0)
        pulse.play(pulse.GaussianSquare(dura12, amp12, sigma12, width12), u0)

    return w
# === Bounds ===
bounds = [
    (60,200),
    (-1,1),
    (-1,1),
    (-1,1),
    
   (60,200),
    (-1,1),
    (-1,1),
    (-1,1),
    
    (60,200),
    (-1,1),
    (-1,1),
    (-1,1),
    
    (60,200),
    (-1,1),
    
    (60,200),
    (-1,1),


    (60,200)
    ,(-1,1)
]

# === Función objetivo ===
iteracion = {"n":0}

def objective_123(x):
    signal = esque_pul_w(x)
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
    
    conc_01 = concurrence(partial_trace(yr, [2]))**2
    conc_02 = concurrence(partial_trace(yr, [1]))**2  
    conc_12 = concurrence(partial_trace(yr, [0]))**2
    conc_function = conc_01 + conc_02 + conc_12
    
    cost = (
        10*(conc_01 - 0.4444444)**2 + 5*(conc_12 - conc_01)**2 + 5*(conc_02 -conc_01)**2 # Igualdad entre concurrencias
        + 5*(np.sqrt(conc_01) + np.sqrt(conc_02) + np.sqrt(conc_12) - 2)**2  # Suma ≈ 2

    )
    
    return np.sqrt(cost)

def callback(xk, convergence):
    iteracion["n"] += 1
    costo = objective_123(xk)
    print(f"{iteracion['n']:3d} | Cost: {costo:.6f} | x: {xk}")

# === Multi-seeds ===
# === Optimización sin seed ===
import json, os

carpeta_salida = "resultados_optimizacion_random"
os.makedirs(carpeta_salida, exist_ok=True)

num_runs = 10   # 👈 número de ejecuciones independientes (cambia según necesites)
resultados = []

for run in range(num_runs):
    iteracion["n"] = 0
    print(f"\n===== Ejecutando run #{run} (sin seed) =====")
    
    result = differential_evolution(
        func=objective_123,
        bounds=bounds,
        callback=callback,
        disp=True,
        maxiter=20,     # 👈 límite de iteraciones
        polish=True     # 👈 refina la solución final con L-BFGS-B
    )
    print(f"✅ Run {run} completado | Costo mínimo: {result.fun}")
    
    # Guardar resultado individual
    resultados_json = {
        "run": run,
        "parametros_optimos": result.x.tolist(),
        "costo_minimo": float(result.fun),
        "iteraciones": iteracion["n"],
        "success": result.success,
        "message": result.message
    }
    nombre_archivo = os.path.join(carpeta_salida, f"resultado_run{run}.json")
    with open(nombre_archivo, "w") as f:
        json.dump(resultados_json, f, indent=4)
    resultados.append(resultados_json)

# === Resumen mejor resultado ===
best = min(resultados, key=lambda x: x["costo_minimo"])
print("\n🏆 Mejor resultado (sin seed):")
print("Run:", best["run"])
print("Costo mínimo:", best["costo_minimo"])
print("Parámetros:", best["parametros_optimos"])

# Guardar resumen
nombre_resumen = os.path.join(carpeta_salida, "resumen_mejor.json")
with open(nombre_resumen, "w") as f:
    json.dump(best, f, indent=4)

print(f"\n📁 Resultados guardados en carpeta '{carpeta_salida}'")
