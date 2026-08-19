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