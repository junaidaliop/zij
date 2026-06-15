# Scalable On-Hardware QNN training (parallelised parameter-shift rule)

Implements the parallelised parameter-shift rule, a hardware-efficient gradient estimator for training quantum neural networks built from commuting-block (Butterfly) layers.

Training a parametrised quantum circuit requires the gradient of the loss with respect to each gate angle, which on hardware is obtained from the parameter-shift rule: evaluating the circuit at shifted parameter values. The naive rule shifts one parameter at a time, so the per-layer cost grows linearly with the number of parameters. The key observation is that within a Butterfly layer the RBS-gate generators $G_j \propto Y_{j_1}\otimes X_{j_2} - X_{j_1}\otimes Y_{j_2}$ mutually commute, and the global measurement $Z^{\otimes n}$ commutes with each generator. The corresponding gradient observables $O_k = [G_k, H]$ are then simultaneously diagonalisable, so every parameter in the layer can be shifted at once and all gradients read out from a constant number of circuit evaluations.

For an RBS gate the four-term shift rule (with shifts $\pi/4$ and $\pi/2$) gives the exact partial derivative; exploiting the commuting structure, all parameters of a layer are shifted simultaneously and the estimate is plugged into a gradient-descent step:

$$
\begin{aligned}
\frac{\partial f(\theta)}{\partial \theta_i} &= \left[ f\!\left(\theta + \tfrac{\pi}{4}\right) - f\!\left(\theta - \tfrac{\pi}{4}\right) \right] - \frac{\sqrt{2}-1}{2}\left[ f\!\left(\theta + \tfrac{\pi}{2}\right) - f\!\left(\theta - \tfrac{\pi}{2}\right) \right], \quad \forall i, \\
\theta_{t+1} &= \theta_t - \eta\, \nabla_\theta f(\theta_t).
\end{aligned}
$$

where $\theta$ are the gate angles, $\eta$ is the learning rate, $f(\theta) = \langle \psi(\theta,x)|\,H\,|\psi(\theta,x)\rangle$ is the measured loss observable, the $\pm\pi/4$ and $\pm\pi/2$ terms are circuit evaluations at the simultaneously shifted angles, and individual gradients are recovered by diagonalising the commuting gradient observables $O_k = [G_k, H]$.

Reference: Natansh Mathur, Panagiotis Kl. Barkoutsos, Masako Yamada, Martin Roetteler, Iordanis Kerenidis, "Scalable On-Hardware Training of Quantum Neural Networks and Application to Clinical Data Imputation", arXiv 2026. https://arxiv.org/abs/2606.03517

---
[Back to the Canon](../README.md)
