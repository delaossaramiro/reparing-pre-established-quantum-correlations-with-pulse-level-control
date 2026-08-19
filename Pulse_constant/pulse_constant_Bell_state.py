###### LLamar paquetes

import numpy as np
from qiskit.quantum_info import partial_trace,Operator, Statevector, negativity, DensityMatrix
from qiskit_dynamics import Solver, Signal
from qiskit_dynamics.models import HamiltonianModel
from qiskit_dynamics.signals import Signal, SignalSum

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from qiskit_ibm_runtime.fake_provider import  FakeSherbrooke
from qiskit_ibm_runtime import QiskitRuntimeService
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
import sympy as sym

#  Llamar backend con la cuenta
service = QiskitRuntimeService(channel="ibm_quantum", token="dbbfe737586d3ed8a382f873c5ee9d4f56fa2a31e4800a589a3829cd3b5caea437430476c6f12ca31428243a76cd0d23b1bd095a13350486060f73a43597688c")

# pedir backend 
backend = service.backend('ibm_sherbrooke')

# propiedades de la Backend
propiedades = backend.properties()
configuraciones = backend.configuration()


# parámetros de los qubits  

# frecuencias por 2 pi
w7 = propiedades.qubit_property(7, name='frequency')[0]*2*np.pi*1e-9
w8 = propiedades.qubit_property(8, name='frequency')[0]*2*np.pi*1e-9
w9 = propiedades.qubit_property(9, name='frequency')[0]*2*np.pi*1e-9


v7 = w7/(2*np.pi)
v8 = w8/(2*np.pi)
v9 = w9/(2*np.pi)

#  acoples
J7_8 =  configuraciones.hamiltonian.get("vars").get("jq7q8")
J8_9 =  configuraciones.hamiltonian.get("vars").get("jq8q9")

# stron drive
d0   =  configuraciones.hamiltonian.get("vars").get("omegad7")
d1   =  configuraciones.hamiltonian.get("vars").get("omegad8")

# tiempo de resolución
dt = backend.dt*1e9


# Función para los pulsos
def create_const_signal(start, duration, amplitude,t):
    return amplitude * ((t >= start) & (t < start + duration))


# negatividad 
def negativi(rho):
    Nega = negativity(DensityMatrix(rho),[0])
    return Nega


# construcción del Hamiltoniano 
dim = 2
a = np.diag(np.sqrt(np.arange(1, dim)), 1)
adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
N = np.diag(np.arange(dim))

ident = np.eye(dim, dtype=complex)
full_ident = np.eye(dim**2, dtype=complex)

N0 = np.kron(ident, N)
N1 = np.kron(N, ident)

a0 = np.kron(ident, a)
a1 = np.kron(a, ident)

a0dag = np.kron(ident, adag)
a1dag = np.kron(adag, ident)

# Haniltoniano CR_q7_q8
static_ham = w7 * N0  + w8 * N1  + J7_8*(a0dag @ a1 +  a0 @ a1dag)

# hamiltonianos de control 
H0_drive78 =  d0 * (a0 + a0dag)
H1_drive78 =  d1 * (a1 + a1dag)

# estado inicial
y0  = Statevector.from_label("00")

# modelo 
model = Solver(static_hamiltonian=static_ham,hamiltonian_operators=[H0_drive78, H1_drive78, H0_drive78])

# función de costo 
# maxima la negatividad 
def cost_function(params, model):
    amp_01, dur_01, amp_11, dur_11, amp_12, amp_20 = params

    # Tiempos relativos
    t1 = (dur_01)*dt
    t2 = (dur_11)*dt
    tt = t1 + t2 
    # sampler 
    time = np.linspace(0, tt, 300)

    # Pulsos
    # pulsos para el canal de control d_0
    signald_0 =  lambda t: create_const_signal(0, t1, amp_01,t)
    signal1 = Signal(signald_0,carrier_freq= v7)
    
    
    # pulso para el canal target d_1
    # hay dos pulsos uno para cada instante.
    signald_1 = SignalSum([
        Signal(lambda t: create_const_signal(0, t1 , amp_11,t), carrier_freq=v8),     # bloque 1
        Signal(lambda t: create_const_signal(t1, t2, amp_12,t), carrier_freq= v8)     # bloque 2
    ])
    
    # pulso para elcanal que controla u_01
    signald_2 = Signal(lambda t: create_const_signal(t1,t2,amp_20,t),carrier_freq= v8)

    # vector inicial.
    y0 = Statevector.from_label("00")
    result = model.solve(t_span=[0,tt], y0=y0, signals=[signal1, signald_1, signald_2],method='jax_odeint', t_eval= time)

    #  vector final 
    rho = result.y[-1]
    # negatividad del vector final. 
    return - negativity(rho,[0])  # queremos maximizar => minimizar negativo

# limpia memoria 
import gc
gc.collect()  # Limpia memoria no usada

# ---------- Función de costo ----------
def cost_function(params, model):
    amp_01, dur_01, amp_11, dur_11, amp_12, amp_20 = params

    # Tiempos y discretización
    t1 = dur_01 * dt
    t2 = dur_11 * dt
    total_time = t1 + t2
    time = np.linspace(0, total_time, 250)

    # Señales
    signal_0 = Signal(lambda t: create_const_signal(0, t1, amp_01, t), carrier_freq=v7)
    signal_1 = SignalSum([
        Signal(lambda t: create_const_signal(0, t1, amp_11, t), carrier_freq=v8),
        Signal(lambda t: create_const_signal(t1, t2, amp_12, t), carrier_freq=v8)
    ])
    signal_2 = Signal(lambda t: create_const_signal(t1, t2, amp_20, t), carrier_freq=v8)

    # Estado inicial
    y0 = Statevector.from_label("00")

    # Simulación
    result = model.solve(t_span=[0, total_time], y0=y0, signals=[signal_0, signal_1, signal_2], 
                         method='jax_odeint', t_eval=time, rtol=1e-4, atol=1e-6)

    rho = result.y[-1]
    try:
        ent = negativity(rho, qargs=[0])
        return -ent  # para maximizar
    except:
        return 1.0  # penaliza si da error

# ---------- Historial para graficar ----------
history = []

def callback(xk):
    val = -cost_function(xk, model)  # valor actual de negatividad
    history.append(val)
    print(f"Iteración {len(history)}: Negatividad = {val:.4f}")

# ---------- Parámetros iniciales ----------
params0 =  [5.15834820e-01, 1.90968973e+02, 6.36226545e-01, 5.88796919e+02 ,4.40746478e-01, 8.94800618e-01]
          

# ---------- Optimización ----------
result = minimize(cost_function, params0, args=(model,), method='Nelder-Mead',
                  options={'maxiter': 1000, 'disp': True}, callback=callback)

# ---------- Resultados ----------
print("\nMejores parámetros encontrados:")
print(result.x)
print(f"Máxima negatividad alcanzada: {-result.fun:.4f}")

# ---------- Gráfico de la evolución ----------
plt.plot(history, marker='o')
plt.xlabel("Iteración")
plt.ylabel("Negatividad")
plt.title("Evolución de la negatividad durante la optimización")
plt.grid(True)
plt.show()
