# SGD (Superpositional Gradient Descent)

Implements SGD (Superpositional Gradient Descent), an Adam update augmented with a quantum-inspired sinusoidal perturbation.

The method keeps the standard Adam adaptive step and adds a perturbation $\mathcal{Q}$ that modulates the gradient by $\sin(\pi\theta_i)$ on the first $n_{\mathrm{qubits}}$ coordinates. The sine modulation mimics the interference patterns of quantum wave functions, producing oscillatory updates that vary smoothly with the parameter value and help the optimizer explore multiple configurations and escape poor local minima. The Adam moments $m_t, v_t$ and constants $\beta_1, \beta_2, \epsilon$ are retained from standard Adam.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\mathcal{Q}(\theta_t, g_t)_i &=
\begin{cases}
\sin(\pi\,\theta_{t,i})\, g_{t,i} & i < n_{\mathrm{qubits}} \\
0 & \text{otherwise}
\end{cases} \\
\theta_{t+1} &= \theta_t - \alpha\left(\frac{m_t}{\sqrt{v_t + \epsilon}} + \lambda\,\mathcal{Q}(\theta_t, g_t)\right)
\end{aligned}
$$

where $\theta$ are the parameters, $\alpha$ the learning rate, $g_t$ the gradient, $m_t$ and $v_t$ the first and second Adam moments with decays $\beta_1, \beta_2$, $\epsilon$ a stability constant, $\lambda$ the quantum weight controlling perturbation strength, and $n_{\mathrm{qubits}}$ the number of leading coordinates that receive the sinusoidal modulation.

Reference: Ahmet Erdem Pamuk, Emir Kaan Özdemir, Şuayp Talha Kocabay, "Superpositional Gradient Descent: Harnessing Quantum Principles for Model Training", IEEE QAI 2025. https://arxiv.org/abs/2511.01918

---
[Back to the Canon](../README.md)
