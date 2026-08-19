""" Genuine multipartite entanglement / correlations for 3-qubit MIXED states. 
=========================================================================

Two measures that (i) work for mixed states, (ii) need N= 3-tangle convex roof, and (iii) are maximixed by the GHZ state


1) -- Genuine Multipartite Negativity (Jungnitsc, Moroder, Guehne,
PRL 106, 190502 (2011)). A proper genuine multipartite entanglement 
monotone: it is ZERO for every biseparable state (pure or mixed,
including mixures of different bipartitions), and positive ONLY when 
the state is genuinely tripartite entangled. Computed by a single 
semidefinnite program (SDP) -- no convex roof. Max over 3 quibits is 
the GHZ state, GMN(GHZ) = 1/2.   


2) pi - tangle residual 3-partite entanglement built from NEGATIVITIES 
 (Ou & Fan, PRA 75, 062308 (2007)). Fully computable for mixed 
 computable cousin of the 3-tangle and is maximized by GHZ
 (pi(GHZ)=1). NOTE: it is a residual entanglement quantifier, not a 
 strict GME monotone, so it can be > 0 for some biseparable state. 
 Use it as a fast GHZ-oriented figure of merit; use GMN when you.
  need a rigorous "genuine" certificate. 
   

"""

import numpy as np
import cvxpy as cp


# ------------------------------------------------------------------
# Basic linear - algebra helpers (pure numpy)
# ------------------------------------------------------------------

def partial_transpose(rho, dims, sys):
    """
    Partial transpose of density matrix rho over subsystem index sys. 
    """
    n = len(dims)
    d = int(np.prod(dims))
    r = rho.reshape(dims + dims)
    exes = list(range(2*n))
    exes[sys], exes[n+sys] = exes[n+sys], exes[sys]
    return r.transpose(exes).reshape(d, d)


def partial_trace(rho, dims, keep):
    """
     Partial trace of `rho` keeping the subsystems listed in `keep`  
    """
    n    = len(dims)
    keep = sorted(keep)
    r    = rho.reshape(dims + dims)
    trace_out = [i for i in range(n) if i not in keep]
    for t in sorted(trace_out, reverse=True):
        r = np.trace(r,axis1=t, axis2=t + (r.ndim//2))
    d_keep = int(np.prod([dims[i] for i in keep]))
    return r.reshape(d_keep, d_keep) 


def negativity(rho, dims, sys):
    """Negativity N = ||rho^{T_sys}||_1 - 1  (Bell/GHZ bipartition -> 1)."""
    pt = partial_transpose(rho, dims, sys)
    pt = (pt +pt.conj().T)/2.0   # symmetrize to avoid numerical issues
    ev = np.linalg.eigvalsh(pt)
    return float(np.sum(np.abs(ev)) - 1.0) 
    
    
# ------------------------------------------------------------------
#1) Genuine Multipartite Negativity (GMN) via SDP
#------------------------------------------------------------------

def genuine_multipartite_negativity(rho,solver=None):
    """ 
    GMN for a 3-qubit density matrix `rho` (8x8, trace 1).    
    Canonical PPT-mixer witness SDP (Jungnitsch-Moroder-Guehne):      
    minimize   Tr(W rho)        subject to for each bipartition m:  W = P_m + Q_m^{T_m},        
    P_m >= 0,   0 <= Q_m <= I   
    and returns  GMN = -min Tr(W rho)   ( >= 0 ; 0 iff biseparable/PPT-mixture ).   
    Normalization: GMN(GHZ) = 1/2,  GMN(W) = sqrt(2)/3 ~ 0.4714.  GHZ is the   
    maximizer over all 3-qubit states.    """
    
    dims = [2, 2, 2]
    d = 8
    rho = np.asarray(rho, dtype= complex)
    
    W = cp.Variable((d,d), hermitian=True)
    constraints = []
    for m in range(3):  # bipartitions A|BC, B|AC, C|AB
         P = cp.Variable((d, d), hermitian=True)        
         Q = cp.Variable((d, d), hermitian=True)
         constraints += [P >> 0, Q >> 0, np.eye(d) - Q >> 0,
                         W == P + cp.partial_transpose(Q, dims, m)]       
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(W @ rho))), constraints)
     # CLARABEL is default; fall back to SCS (higher precision) if needed.  
    order = [solver] if solver else [cp.CLARABEL, cp.SCS]
    val = None
    for s in order:
        try:
             prob.solve(solver=s)
             if prob.value is not None:
                  val = prob.value
                  if prob.status == cp.OPTIMAL:
                       break
        except cp.error.SolverError:
             continue
    if val is None:
               raise RuntimeError("SDP did not solve; install another solver (e.g. MOSEK).")
    return max(0.0, -val)
                      
    


# ------------------------------------------------------------------
# 2) pi- tangle (negativity-based residual 3-tangle) 
#-------------------------------------------------------------------

def pi_tangle(rho):
     """Ou-Fan pi-tangle for a 3-qubit density matrix. pi(GHZ)=1."""
     dims = [2,2,2]
     rho = np.asarray(rho, dtype= complex)
     
     # one-vs-rest negativities
     N_A = negativity(rho, dims, 0) # A|BC
     N_B = negativity(rho, dims, 1) # B|AC
     N_C = negativity(rho, dims, 2) # C|AB
     
     # two-qubit reduced negativities
     rho_AB = partial_trace(rho, dims, keep=[0, 1])
     rho_AC = partial_trace(rho, dims, keep=[0, 2])
     rho_BC = partial_trace(rho, dims, keep=[1, 2])
     
     N_AB = negativity(rho_AB, [2, 2], 0)
     N_AC = negativity(rho_AC, [2, 2], 0)
     N_BC = negativity(rho_BC, [2, 2], 0)
     
     pi_A = N_A ** 2 - N_AB ** 2 - N_AC ** 2
     pi_B = N_B ** 2 - N_AB ** 2 - N_BC ** 2
     pi_C = N_C ** 2 - N_AC ** 2 - N_BC ** 2
     
     return (pi_A + pi_B + pi_C) / 3.0
 
 
 #--------------------------------------------------------
 # Demo / self-tests
#----------------------------------------------------------

def _ket(bits):
    V = np.zeros(8, dtype=complex)
    V[int(bits,2)] = 1.0
    return V

if __name__ == "__main__":
    # GHZ state
    ghz = (_ket("000") + _ket("111")) / np.sqrt(2)
    rho_ghz = np.outer(ghz, ghz.conj())
    
    # W
    w = (_ket("001") + _ket("010") + _ket("100")) / np.sqrt(3)
    rho_w = np.outer(w, w.conj())
    
    
    # fully separable product |+>|+>|+>? use |000>
    rho_prod = np.outer(_ket("000"), _ket("000").conj())
    
    
    # biseparable mixed: (Bell on AB)⊗|0>_C  mixed with (Bell on AC)⊗|0>_B
    bell_ab = (_ket("000") + _ket("110")) / np.sqrt(2)
    bell_ac = (_ket("000") + _ket("101")) / np.sqrt(2)
    r_ab = np.outer(bell_ab, bell_ab.conj())
    r_ac = np.outer(bell_ac, bell_ac.conj())
    rho_bisep = 0.5 * r_ab + 0.5 * r_ac
    
    # GHZ + white noise:  p |GHZ><GHZ| + (1-p) I/8
    def ghz_noisy(p):
        return p * rho_ghz + (1 - p) * np.eye(8) / 8
    
    print("state            GMN        pi-tangle")
    print("-" * 42)
    for name, r in [("GHZ", rho_ghz), ("W", rho_w),
                    ("product |000>", rho_prod),
                    ("biseparable mix", rho_bisep)]:
        print(f"{name:16s} {genuine_multipartite_negativity(r):8.4f}   {pi_tangle(r):8.4f}")
    print("\nGHZ + white noise  (genuinely entangled iff p > 3/7 = 0.4286):")
    for p in [0.3, 3/7, 0.5, 0.7, 1.0]:
        r = ghz_noisy(p)
        print(f"  p={p:5.3f}   GMN={genuine_multipartite_negativity(r):8.4f} pi= {pi_tangle(r):7.4f}")
    
    
        
                    
    
from gme_3qubits import genuine_multipartite_negativity,pi_tangle
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix,negativity,state_fidelity,Operator,concurrence,entropy,purity
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


from qiskit_dynamics import Solver, Signal

from qiskit import pulse
import matplotlib.pyplot as plt
# from scipy.optimize import minimize
import numpy as np
# import sympy as sym

import jax
jax.config.update("jax_enable_x64", True)

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

# tiempo de relajación y decoherencia en segundos
T1_7 = propiedades.qubit_property(7,'T1')[0]  # tiempo de relajación
T2_7 = propiedades.qubit_property(7,'T2')[0]  # tiempo de decoherencia
T1_8 = propiedades.qubit_property(8,'T1')[0]  # tiempo de relajación
T2_8 = propiedades.qubit_property(8,'T2')[0]  # tiempo de decoherencia
T1_9 = propiedades.qubit_property(9,'T1')[0]  # tiempo de relajación    
T2_9 = propiedades.qubit_property(9,'T2')[0]  # tiempo de decoherencia  


v7 = w7/(2*np.pi)  # frecuencia en GHz
v8 = w8/(2*np.pi)  # frecuencia en GHz
v9 = w9/(2*np.pi)  # frecuencia en GHz  


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
full_ident = np.eye(dim**3, dtype=complex)

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
  

# Tiempos T1 y T2 en ns
T1_7 = T1_7*1e9
T2_7 = T2_7*1e9
T1_8 = T1_8*1e9
T2_8 = T2_8*1e9
T1_9 = T1_9*1e9
T2_9 = T2_9*1e9
# gammas de relajación y decoherencia
gamma1_7 = 1/T1_7
gamma2_7 = 1/T2_7 - 1/(2*T1_7)
gamma1_8 = 1/T1_8
gamma2_8 = 1/T2_8 - 1/(2*T1_8)
gamma1_9 = 1/T1_9
gamma2_9 = 1/T2_9 - 1/(2*T1_9)


Z0  = full_ident - 2*N0
Z1  = full_ident - 2*N1
Z2  = full_ident - 2*N2

sm1 = a0
sm2 = a1
sm3 = a2

sz1 = Z0
sz2 = Z1 
sZ3 = Z2

collapse_operators = [np.sqrt(gamma1_7)*a0,
                      np.sqrt(gamma1_8)*a1,
                      np.sqrt(gamma1_9)*a2,
                      np.sqrt(gamma2_7)*Z0,
                      np.sqrt(gamma2_8)*Z1,
                      np.sqrt(gamma2_9)*Z2]



from qiskit import pulse

def esque_pul(params):
    t_0, t_1, t_2,t_3,A0, A1, A2, A3, A4,  = params

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
        # pulse.play(pulse.Constant(duration=t_3, amp=A4),d2)
        
        
      # pulso u1
        pulse.play(pulse.Constant(duration=t_3, amp=A4),u1)
        
    return kp


solver = Solver(static_hamiltonian=static_ham,
                hamiltonian_operators=[H_drive7, H_drive8, H_drive9, H_drive7, H_drive8],
                static_dissipators = collapse_operators,
                dissipator_channels= None,
                hamiltonian_channels= ["d0", "d1", "d2", "u0", "u1"],
                channel_carrier_freqs= {"d0": v7, "d1": v8, "d2": v9, "u0": v8, "u1": v9},
                dt=dt,
                array_library='jax'
                )





rho_0 = DensityMatrix.from_label('000')


X_GHZ =  [
    1.91779586e+02, 1.03750354e+03, 1.68736170e+02, 1.45046248e+03,
    4.90394435e-01, -6.91612520e-01, 6.65492967e-01, 1.65200193e-01,
    6.19735822e-01
]


def evol_GHZ_lindblab(x):
    signal = esque_pul(x)
    tiempo_t = signal.duration  # duración total del pulso en pasos de dt
    sol = solver.solve(
        t_span=[0, tiempo_t*dt],
        y0=rho_0,
        signals=signal,
        # method="jax_odeint",
        atol=1e-7,
        rtol=1e-8
    )
    # yf = sol.y[-1]
    
    return sol


matrix_GHZ = evol_GHZ_lindblab(X_GHZ)


import numpy as np
from qiskit.quantum_info import DensityMatrix

def sanitize_density_matrix(rho):
    """
    Corrige una matriz para que sea una matriz densidad física:
    - Hermítica
    - Traza 1
    - Semidefinida positiva
    """
    rho = np.array(rho, dtype=complex)

    # 1) Forzar hermiticidad
    rho = 0.5 * (rho + rho.conj().T)

    # 2) Diagonalizar
    eigvals, eigvecs = np.linalg.eigh(rho)

    # 3) Eliminar negativos pequeños
    eigvals[eigvals < 0] = 0

    # 4) Reconstruir
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

    # 5) Renormalizar traza
    rho = rho / np.trace(rho)

    return DensityMatrix(rho)


g_m_n = []
time_GHZ_open = []
for k, j in enumerate(matrix_GHZ.y):
    g_m_n.append(genuine_multipartite_negativity(sanitize_density_matrix(j.data)))
    time_GHZ_open.append(matrix_GHZ.t[k])
    print(genuine_multipartite_negativity(sanitize_density_matrix(j.data)))



# 2. Crear el gráfico
plt.plot(time_GHZ_open, g_m_n, marker='o', linestyle='-', color='b', label='3_tangle sistema abierto')

# 3. Personalizar (opcional)
plt.title("genuine_multipartite_negativity")
plt.xlabel("tiempo ns")
plt.ylabel("3-tangle")
plt.legend()
plt.grid(True)

# 4. Mostrar el gráfico
plt.show()


# Guardar
np.savez(
    "GHZ_open.npz",
    time=time_GHZ_open,
    GMN=g_m_n
)
