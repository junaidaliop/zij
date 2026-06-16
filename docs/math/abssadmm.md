# AbsSADMM

Implements AbsSADMM, a stochastic ADMM for nonconvex nonsmooth problems whose mini-batch size adapts to the progress of the iterates.

AbsSADMM solves $\min_{x,y} f(x) + g(y)$ subject to $Ax + By = c$, where $f$ is a smooth finite-sum and $g$ is nonsmooth. Each iteration linearizes the smooth part with a mini-batch gradient and takes a proximal step in $x$, an augmented-Lagrangian minimization in $y$, and a dual ascent in $\lambda$. The novelty is the batch size $M_k$: it scales inversely with the squared distance between consecutive iterates, so the method uses small batches when the iterates move a lot (early on) and large batches as they settle, while a second term caps the batch at the level dictated by the target accuracy $\epsilon$.

With the augmented Lagrangian $\mathcal{L}_\beta(x,y,\lambda) = f(x) + g(y) - \lambda^\top(Ax + By - c) + \tfrac{\beta}{2}\lVert Ax + By - c\rVert^2$ and the proximal regularizer chosen as $G = rI - \beta\eta A^\top A$, one step reads:

$$
\begin{aligned}
M_k &= \min\left\{ c_\tau\, \sigma^2\, \lVert x_k - x_{k-1}\rVert^{-2},\ c_\varepsilon\, \sigma^2\, \epsilon^{-1} \right\} \\
y_{k+1} &= \arg\min_{y}\ \mathcal{L}_\beta(x_k, y, \lambda_k) \\
x_{k+1} &= x_k - \frac{\eta}{r}\left( \nabla f_{I_k}(x_k) + \beta A^\top\!\left( Ax_k + By_{k+1} - c - \tfrac{\lambda_k}{\beta} \right) \right) \\
\lambda_{k+1} &= \lambda_k - \beta\left( Ax_{k+1} + By_{k+1} - c \right)
\end{aligned}
$$

where $x,y$ are the primal blocks, $\lambda$ the dual variable, $\eta$ the step size, $\beta$ the penalty parameter, $r$ the proximal coefficient, $\nabla f_{I_k}(x_k) = \tfrac{1}{|I_k|}\sum_{i\in I_k}\nabla f_i(x_k)$ the mini-batch gradient with $|I_k| = M_k$, $\sigma^2$ the gradient-variance bound, $c_\tau, c_\varepsilon$ positive constants, and $\epsilon$ the target accuracy.

Reference: Jiachen Jin, Kangkang Deng, Boyu Wang, Hongxia Wang, "Stochastic ADMM with batch size adaptation for nonconvex nonsmooth optimization", arXiv 2025. https://arxiv.org/abs/2505.06921

---
[Back to the Canon](../index.md)
