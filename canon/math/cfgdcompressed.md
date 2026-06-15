# CFGD (Compressed)

Implements compressed FGD (CFGD), a stochastic fractional gradient method that scales and compresses the fractional gradient with a matrix-valued stepsize.

Fractional gradient descent replaces the integer first derivative with a Caputo-style fractional gradient $\delta^{\alpha} f$, whose entries weight the running displacement to a reference point by $1/\Gamma(2-\alpha)$. CFGD carries this idea into the matrix-stepsize compressed setting of det-CGD: instead of a scalar learning rate it uses a positive-definite matrix stepsize $D$, and an unbiased random sketch $S^k$ (with $\mathbb{E}[S^k]=I$) compresses the step so it can be communicated cheaply in a distributed or federated run. Because $D$ adapts to the matrix-smoothness structure of the objective, the matrix stepsize captures curvature that a scalar step cannot, giving faster convergence on matrix-smooth non-convex problems. Two variants differ only in the order of the stepsize and the sketch.

$$
\begin{aligned}
\text{CFGD}_1:\quad \theta_{k+1} &= \theta_k - D\,S^k\,\delta^{\alpha} f(\theta_k), \\
\text{CFGD}_2:\quad \theta_{k+1} &= \theta_k - S^k\,D\,\delta^{\alpha} f(\theta_k),
\end{aligned}
$$

where $\theta_k$ are the parameters, $\delta^{\alpha} f(\theta_k)$ is the (Caputo-based) fractional gradient of order $\alpha\in(0,1)$ in place of $\nabla f$, $D\succ 0$ is the fixed matrix stepsize, $S^k$ is a random sketch (compression) matrix that is positive semidefinite and unbiased, $\mathbb{E}[S^k]=I$, and $\Gamma(\cdot)$ is the gamma function entering the fractional-gradient weighting. The two updates coincide when $D$ and $S^k$ commute.

Reference: Alokendu Mazumder, Kshitij Vyas, Punit Rathore, "Fractional Gradient Descent With Matrix Stepsizes for Non-Convex Optimization", IEEE Transactions on Neural Networks and Learning Systems 2025. https://doi.org/10.1109/TNNLS.2025.3637535

---
[Back to the Canon](../README.md)
