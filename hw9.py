import numpy as np

def power_method(A, max_iter=1000, tol=1e-10):
    n = A.shape[0]
    x = np.ones(n)
    x = x / np.linalg.norm(x)
    
    for _ in range(max_iter):
        x_new = A @ x
        x_new = x_new / np.linalg.norm(x_new)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    lambda_est = x.T @ A @ x
    return lambda_est, x

def deflate(A, eigval, eigvec):
    eigvec = eigvec / np.linalg.norm(eigvec)
    A_deflated = A - eigval * np.outer(eigvec, eigvec)
    return A_deflated

# Original matrix
A = np.array([[2,1,1],
              [1,2,1],
              [1,1,2]], dtype=float)

eigenvalues = []
A_current = A.copy()

for i in range(3):  # We know A is 3x3
    lam, vec = power_method(A_current)
    eigenvalues.append(lam)
    A_current = deflate(A_current, lam, vec)

print("Estimated eigenvalues:")
for i, val in enumerate(eigenvalues, 1):
    print(f"λ{i} ≈ {val:.6f}")
