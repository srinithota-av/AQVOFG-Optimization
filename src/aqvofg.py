import os
import numpy as np
from scipy.special import gamma

def aqvofg_optimizer(
    f, 
    grad_f, 
    x0, 
    lr=0.01, 
    alpha0=0.55, 
    beta=0.35, 
    lam=0.10, 
    eta=0.20, 
    omega=2.0, 
    tol=1e-8, 
    max_iter=1000, 
    memory_length=10
):
    """
    Adaptive Quantum Variable-Order Fractional Gradient Optimization (AQVOFG)
    
    Parameters:
    -----------
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient function grad_f(x) returning an array of shape (n,).
    x0 : array-like
        Initial point array of shape (n,).
    lr : float
        Learning rate gamma (default: 0.01).
    alpha0 : float
        Base fractional order (0 < alpha0 < 1).
    beta : float
        Adaptive coefficient for fractional order (0 < beta < 1 - alpha0).
    lam : float
        Exponential decay rate lambda for memory kernel.
    eta : float
        Quantum amplitude modulation parameter (0 <= eta < 1).
    omega : float
        Oscillation frequency parameter.
    tol : float
        Tolerance epsilon for stopping criterion based on AQVOFG norm.
    max_iter : int
        Maximum number of iterations.
    memory_length : int
        Window size 'm' for discrete memory integration.
        
    Returns:
    --------
    x_opt : ndarray
        Optimal point found.
    history : list
        History of objective function values per iteration.
    """
    x = np.array(x0, dtype=np.float64)
    history = [f(x)]
    
    # Stores past states and gradients: list of tuples (x_s, grad_s)
    history_states = []

    for k in range(max_iter):
        # 1. Compute current gradient
        g_k = grad_f(x)
        norm_gk = np.linalg.norm(g_k)
        
        # Save current state to memory buffer
        history_states.append((x.copy(), g_k.copy()))
        if len(history_states) > memory_length:
            history_states.pop(0)

        # 2. Compute adaptive fractional order alpha_k (Equation 1)
        alpha_k = alpha0 + beta * (norm_gk / (1.0 + norm_gk))

        # 3. Compute fractional weighting factor 1 / Gamma(1 - alpha_k)
        gamma_factor = 1.0 / gamma(1.0 - alpha_k)

        # 4. Construct adaptive quantum memory kernel and accumulate over memory window (Equations 2 & 3)
        AQVO_grad = np.zeros_like(x)
        
        for s_idx, (x_s, grad_s) in enumerate(history_states):
            dist = np.linalg.norm(x - x_s)
            eps = 1e-8
            
            # Quantum phase angle theta(x) = arctan(||grad f(x)||)
            theta_x = np.arctan(norm_gk)
            
            # Memory Kernel K(x, s) (Equation 2)
            kernel = (
                np.exp(-lam * dist) 
                * (1.0 + eta * np.sin(omega * k)) 
                * (np.cos(theta_x) ** 2)
            )
            
            # Fractional denominator (||x - s||)^{\alpha(x)}
            denom = (dist + eps) ** alpha_k
            
            # Accumulate integrand
            AQVO_grad += kernel * grad_s / denom

        # Apply Gamma scaling factor
        G_k = gamma_factor * AQVO_grad

        # 5. Convergence check
        if np.linalg.norm(G_k) < tol:
            break

        # 6. Update optimization variable (Equation 4 & Algorithm 1)
        x = x - lr * G_k
        history.append(f(x))

    return x, history


# ==========================================
# Benchmark Test Functions & Runner
# ==========================================

def sphere_f(x):
    return np.sum(x**2)

def sphere_grad(x):
    return 2.0 * x

if __name__ == "__main__":
    print("Running AQVOFG Optimization Test...")
    
    # Problem settings based on Paper Table 1 & Table 2
    dim = 30
    x0 = np.ones(dim) * 5.0  # Initial point
    
    # Run Optimizer
    x_opt, fit_history = aqvofg_optimizer(
        f=sphere_f,
        grad_f=sphere_grad,
        x0=x0,
        lr=0.01,
        alpha0=0.55,
        beta=0.35,
        lam=0.10,
        eta=0.20,
        omega=2.0,
        tol=1e-8,
        max_iter=1000,
        memory_length=10
    )

    print(f"Optimal Value Found: {sphere_f(x_opt):.4e}")
    print(f"Total Iterations: {len(fit_history) - 1}")
    
    # Ensure data output directory exists and export raw results CSV
    results_dir = os.path.join("data", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, "convergence_curves.csv")
    with open(csv_path, "w") as f:
        f.write("iteration,AQVOFG\n")
        for idx, val in enumerate(fit_history):
            f.write(f"{idx},{val:.6e}\n")
            
    print(f"Saved convergence results to '{csv_path}'")
