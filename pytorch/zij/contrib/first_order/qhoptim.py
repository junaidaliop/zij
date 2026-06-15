# Adapted from https://github.com/facebookresearch/qhoptim (commit e81dea3)
# Copyright (c) Facebook, Inc. and its affiliates. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the quasi-hyperbolic optimizers QHM and QHAdam."""

import collections
import math

import torch

from ...core.optimizer import Optimizer, required

__all__ = ["QHM", "QHAdam", "QHAdamW"]


QHMParams = collections.namedtuple("QHMParams", ["alpha", "nu", "beta"])

QHAdamParams = collections.namedtuple("QHAdamParams", ["alpha", "nu1", "nu2", "beta1", "beta2"])


def _from_pid(k_p, k_i, k_d):
    alpha = k_i
    nu = k_p * k_p / (k_i * k_d)
    beta = k_d / (k_d - k_p)
    return QHMParams(alpha=alpha, nu=nu, beta=beta)


def _from_synthesized_nesterov(alpha, beta1, beta2):
    new_alpha = alpha / (1.0 - beta1)
    nu = 1.0 - ((1.0 - beta1) / beta1) * beta2
    beta = beta1
    return QHMParams(alpha=new_alpha, nu=nu, beta=beta)


def _from_robust_momentum(l, kappa, rho):
    if rho is None:
        rho = 1.0 - 1.0 / math.sqrt(kappa)

    alpha = kappa * ((1.0 - rho) ** 2) * (1.0 + rho) / l
    beta1 = kappa * (rho ** 3) / (kappa - 1.0)
    beta2 = (rho ** 3) / ((kappa - 1.0) * ((1.0 - rho) ** 2) * (1.0 + rho))
    return _from_synthesized_nesterov(alpha, beta1, beta2)


def _from_accsgd(delta, kappa, xi, eps):
    alpha = (delta * eps * (1.0 + xi)) / (1.0 + eps)
    nu = (eps * xi - 1.0) / (eps * (1.0 + xi))
    beta = (kappa - (eps * eps) * xi) / (kappa + eps * xi)
    return QHMParams(alpha=alpha, nu=nu, beta=beta)


def _from_two_state_optimizer(h, k, l, m, q, z):
    phi = math.sqrt((h - q) * (h - q) + 4.0 * k * m)
    psi = k * m - h * q
    xi = (h - q - phi) * (l * m - h * z) + 2.0 * m * (l * q - k * z)

    alpha = 0.5 * xi / (phi * psi)
    nu = 2.0 * m * (l * q - k * z) / xi
    beta = 0.5 * (h + q - phi)
    return QHMParams(alpha=alpha, nu=nu, beta=beta)


def _from_nadam(lr, beta1, beta2):
    return QHAdamParams(alpha=lr, nu1=beta1, nu2=1.0, beta1=beta1, beta2=beta2)


class QHM(Optimizer):
    r"""Implements quasi-hyperbolic momentum (QHM), a discounted interpolation
    between plain SGD and momentum.

    .. math::
       \begin{aligned}
          g_{t+1} &= \beta\, g_t + (1 - \beta)\, \nabla_t \\
          \theta_{t+1} &= \theta_t
              - \alpha \left[ (1 - \nu)\, \nabla_t + \nu\, g_{t+1} \right]
       \end{aligned}

    where :math:`\alpha` is the learning rate, :math:`\beta` the momentum
    factor, :math:`\nu` the immediate discount factor that interpolates between
    plain SGD (:math:`\nu = 0`) and momentum (:math:`\nu = 1`), :math:`g_t` the
    momentum buffer, and :math:`\nabla_t` the gradient at :math:`\theta_t`.

    Note:
        QHM uses dampened momentum. When converting from plain momentum to QHM,
        scale the learning rate by :math:`1 / (1 - \beta)`: momentum with
        :math:`\alpha = 0.1` and :math:`\beta = 0.9` corresponds to QHM with
        :math:`\alpha = 1.0`.

    Reference: Jerry Ma, Denis Yarats, "Quasi-hyperbolic momentum and Adam for
    deep learning", ICLR 2019.
    https://arxiv.org/abs/1810.06801
    """

    def __init__(self, params, lr=required, momentum=required, nu=required, weight_decay=0.0, weight_decay_type="grad"):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if weight_decay_type not in ("grad", "direct"):
            raise ValueError("Invalid weight_decay_type value: {}".format(weight_decay_type))

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nu": nu,
            "weight_decay": weight_decay,
            "weight_decay_type": weight_decay_type,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional):
                A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, nu, momentum = group["lr"], group["nu"], group["momentum"]
            weight_decay, weight_decay_type = group["weight_decay"], group["weight_decay_type"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                param_state = self.state[p]

                if weight_decay != 0:
                    if weight_decay_type == "grad":
                        d_p = d_p.add(p, alpha=weight_decay)
                    elif weight_decay_type == "direct":
                        p.mul_(1.0 - lr * weight_decay)

                if len(param_state) == 0:
                    param_state["momentum_buffer"] = torch.zeros_like(p)

                momentum_buffer = param_state["momentum_buffer"]
                momentum_buffer.mul_(momentum).add_(d_p, alpha=1.0 - momentum)

                p.add_(momentum_buffer, alpha=-lr * nu)
                p.add_(d_p, alpha=-lr * (1.0 - nu))

        return loss

    @classmethod
    def _params_to_dict(cls, params):
        return {"lr": params.alpha, "nu": params.nu, "momentum": params.beta}

    @classmethod
    def from_pid(cls, k_p, k_i, k_d):
        r"""Calculates the QHM hyperparameters required to recover a PID
        optimizer as described in `Recht (2018)`_.

        Args:
            k_p (float): proportional gain (see reference)
            k_i (float): integral gain (see reference)
            k_d (float): derivative gain (see reference)

        Returns:
            Three-element ``dict`` containing ``lr``, ``momentum``, and ``nu``
            to use in QHM.

        .. _`Recht (2018)`: https://web.archive.org/web/20181027184056/http://www.argmin.net/2018/04/19/pid/
        """
        return cls._params_to_dict(_from_pid(k_p, k_i, k_d))

    @classmethod
    def from_synthesized_nesterov(cls, alpha, beta1, beta2):
        r"""Calculates the QHM hyperparameters required to recover the
        synthesized Nesterov optimizer (Section 6 of `Lessard et al. (2016)`_).

        Args:
            alpha (float): learning rate
            beta1 (float): first momentum (see reference)
            beta2 (float): second momentum (see reference)

        Returns:
            Three-element ``dict`` containing ``lr``, ``momentum``, and ``nu``
            to use in QHM.

        .. _`Lessard et al. (2016)`: https://arxiv.org/abs/1408.3595
        """
        return cls._params_to_dict(_from_synthesized_nesterov(alpha, beta1, beta2))

    @classmethod
    def from_robust_momentum(cls, l, kappa, rho=None):
        r"""Calculates the QHM hyperparameters required to recover the Robust
        Momentum `(Cyrus et al., 2018)`_ or Triple Momentum
        `(Scoy et al., 2018)`_ optimizers.

        Args:
            l (float): Lipschitz constant of gradient (see reference)
            kappa (float): condition ratio (see reference)
            rho (float, optional): noise-free convergence rate. If None, will
                return the parameters for the Triple Momentum optimizer.

        Returns:
            Three-element ``dict`` containing ``lr``, ``momentum``, and ``nu``
            to use in QHM.

        .. _`(Cyrus et al., 2018)`: https://arxiv.org/abs/1710.04753

        .. _`(Scoy et al., 2018)`: http://www.optimization-online.org/DB_FILE/2017/03/5908.pdf
        """
        return cls._params_to_dict(_from_robust_momentum(l, kappa, rho))

    @classmethod
    def from_accsgd(cls, delta, kappa, xi, eps=0.7):
        r"""Calculates the QHM hyperparameters required to recover the AccSGD
        optimizer `(Kidambi et al., 2018)`_.

        Args:
            delta (float): short step (see reference)
            kappa (float): long step parameter (see reference)
            xi (float): statistical advantage parameter (see reference)
            eps (float, optional): arbitrary value, between 0 and 1 exclusive
                (see reference) (default: 0.7)

        Returns:
            Three-element ``dict`` containing ``lr``, ``momentum``, and ``nu``
            to use in QHM.

        .. _`(Kidambi et al., 2018)`: https://arxiv.org/abs/1803.05591
        """
        return cls._params_to_dict(_from_accsgd(delta, kappa, xi, eps))

    @classmethod
    def from_two_state_optimizer(cls, h, k, l, m, q, z):
        r"""Calculates the QHM hyperparameters required to recover the
        following two-state optimizer (named "TSO" in `Ma and Yarats (2019)`_):

        .. math::
           \begin{aligned}
              a_{t+1} &= h\, a_t + k\, \theta_t + l\, \nabla_t \\
              \theta_{t+1} &= m\, a_t + q\, \theta_t + z\, \nabla_t
           \end{aligned}

        Here :math:`a_t` and :math:`\theta_t` are the two states and
        :math:`\nabla_t` is the gradient with respect to :math:`\theta_t`. Be
        careful that the coefficients satisfy the regularity conditions from
        the reference.

        Returns:
            Three-element ``dict`` containing ``lr``, ``momentum``, and ``nu``
            to use in QHM.

        .. _`Ma and Yarats (2019)`: https://arxiv.org/abs/1810.06801
        """
        return cls._params_to_dict(_from_two_state_optimizer(h, k, l, m, q, z))


class QHAdam(Optimizer):
    r"""Implements QHAdam, the quasi-hyperbolic counterpart of Adam.

    .. math::
       \begin{aligned}
          m_t &= \beta_1\, m_{t-1} + (1 - \beta_1)\, g_t \\
          v_t &= \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2 \\
          \hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
          \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
          \theta_t &= \theta_{t-1} - \alpha\,
              \frac{(1 - \nu_1)\, g_t + \nu_1\, \hat{m}_t}
                   {\sqrt{(1 - \nu_2)\, g_t^2 + \nu_2\, \hat{v}_t} + \epsilon}
       \end{aligned}

    where :math:`\alpha` is the learning rate, :math:`\beta_1, \beta_2` the
    moment decay rates, and :math:`\nu_1, \nu_2` the immediate discount factors
    that interpolate each moment estimate toward the current gradient. Setting
    :math:`\nu_1 = \nu_2 = 1` recovers Adam. The NAdam optimizer is recovered
    through :func:`from_nadam`.

    Reference: Jerry Ma, Denis Yarats, "Quasi-hyperbolic momentum and Adam for
    deep learning", ICLR 2019.
    https://arxiv.org/abs/1810.06801
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        nus=(1.0, 1.0),
        weight_decay=0.0,
        decouple_weight_decay=False,
        eps=1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = {
            "lr": lr,
            "betas": betas,
            "nus": nus,
            "weight_decay": weight_decay,
            "decouple_weight_decay": decouple_weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional):
                A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            nu1, nu2 = group["nus"]
            weight_decay = group["weight_decay"]
            decouple_weight_decay = group["decouple_weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                if d_p.is_sparse:
                    raise RuntimeError("QHAdam does not support sparse gradients")

                param_state = self.state[p]

                if weight_decay != 0:
                    if decouple_weight_decay:
                        p.mul_(1 - lr * weight_decay)
                    else:
                        d_p = d_p.add(p, alpha=weight_decay)

                d_p_sq = d_p.mul(d_p)

                if len(param_state) == 0:
                    param_state["beta1_weight"] = 0.0
                    param_state["beta2_weight"] = 0.0
                    param_state["exp_avg"] = torch.zeros_like(p)
                    param_state["exp_avg_sq"] = torch.zeros_like(p)

                param_state["beta1_weight"] = 1.0 + beta1 * param_state["beta1_weight"]
                param_state["beta2_weight"] = 1.0 + beta2 * param_state["beta2_weight"]

                beta1_weight = param_state["beta1_weight"]
                beta2_weight = param_state["beta2_weight"]
                exp_avg = param_state["exp_avg"]
                exp_avg_sq = param_state["exp_avg_sq"]

                beta1_adj = 1.0 - (1.0 / beta1_weight)
                beta2_adj = 1.0 - (1.0 / beta2_weight)
                exp_avg.mul_(beta1_adj).add_(d_p, alpha=1.0 - beta1_adj)
                exp_avg_sq.mul_(beta2_adj).add_(d_p_sq, alpha=1.0 - beta2_adj)

                avg_grad = exp_avg.mul(nu1)
                if nu1 != 1.0:
                    avg_grad.add_(d_p, alpha=1.0 - nu1)

                avg_grad_rms = exp_avg_sq.mul(nu2)
                if nu2 != 1.0:
                    avg_grad_rms.add_(d_p_sq, alpha=1.0 - nu2)
                avg_grad_rms.sqrt_()
                if eps != 0.0:
                    avg_grad_rms.add_(eps)

                p.addcdiv_(avg_grad, avg_grad_rms, value=-lr)

        return loss

    @classmethod
    def _params_to_dict(cls, params):
        return {"lr": params.alpha, "nus": (params.nu1, params.nu2), "betas": (params.beta1, params.beta2)}

    @classmethod
    def from_nadam(cls, lr=1e-3, betas=(0.9, 0.999)):
        r"""Calculates the QHAdam hyperparameters required to recover the NAdam
        optimizer `(Dozat, 2016)`_.

        This is not an identical recovery of the formulation in the paper, due
        to subtle differences in the application of the bias correction in the
        first moment estimator. In practice, this difference is almost
        certainly irrelevant.

        Args:
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for
                computing running averages of the gradient and its square
                (default: (0.9, 0.999))

        Returns:
            Three-element ``dict`` containing ``lr``, ``betas``, and ``nus``
            to use in QHAdam.

        .. _`(Dozat, 2016)`: https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        """
        return cls._params_to_dict(_from_nadam(lr, betas[0], betas[1]))


def QHAdamW(params, *args, **kwargs):
    r"""Constructs the decoupled weight decay variant of QHAdam, as proposed by
    Loschilov and Hutter (2017).

    Shares all arguments of the :class:`QHAdam` constructor; equivalent to
    constructing :class:`QHAdam` with ``decouple_weight_decay=True``.
    """
    return QHAdam(params, *args, decouple_weight_decay=True, **kwargs)
