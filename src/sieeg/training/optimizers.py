"""Optimisers. The paper trains with Nadam(lr=0.002, beta_1=0.9, beta_2=0.999) (Sec. 5.3).

lr=0.002 is the default of the standalone Keras-2 Nadam, which warms the
momentum up with ``schedule_decay=0.004``. Keras 3 dropped ``schedule_decay``
(its schedule is ``0.96**t`` instead of ``0.96**(0.004 t)``, about 250x faster),
so :class:`LegacyNadam` re-implements the Keras 2.2.4 update (decision D12).
"""

from __future__ import annotations

import keras
from keras import ops

from ..config import TrainConfig


@keras.saving.register_keras_serializable(package="sieeg")
class LegacyNadam(keras.optimizers.Optimizer):
    """Keras 2.2.4 Nadam (Dozat 2016, with ``schedule_decay``). With ``t = iterations + 1``::

        mu_t   = beta_1 * (1 - 0.5 * 0.96 ** (t * schedule_decay))
        mu_t1  = beta_1 * (1 - 0.5 * 0.96 ** ((t + 1) * schedule_decay))
        prod_t = mu_1 * ... * mu_t                       (kept per variable)
        m = beta_1 m + (1 - beta_1) g;   v = beta_2 v + (1 - beta_2) g^2
        m_bar = (1 - mu_t) g / (1 - prod_t) + mu_t1 m / (1 - prod_t mu_t1)
        p -= lr * m_bar / (sqrt(v / (1 - beta_2 ** t)) + epsilon)

    ``schedule_decay=1`` gives the momentum schedule of ``keras.optimizers.Nadam``.
    """

    def __init__(
        self,
        learning_rate=0.002,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        schedule_decay: float = 0.004,
        name: str = "legacy_nadam",
        **kwargs,
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        # Per-variable slots via add_variable_from_reference (every Keras 3), not add_optimizer_variables
        # (Keras >= 3.10). Same names and momentum/velocity interleaving as the latter.
        self._momentums, self._velocities = [], []
        for var in var_list:
            self._momentums.append(self.add_variable_from_reference(var, name="momentum"))
            self._velocities.append(self.add_variable_from_reference(var, name="velocity"))
        self._mu_products = [
            self.add_variable(shape=(), initializer="ones", dtype=var.dtype, name="mu_product") for var in var_list
        ]

    def update_step(self, gradient, variable, learning_rate):
        dtype = variable.dtype
        lr = ops.cast(learning_rate, dtype)
        g = ops.cast(gradient, dtype)
        t = ops.cast(self.iterations + 1, dtype)
        beta_1 = ops.cast(self.beta_1, dtype)
        beta_2 = ops.cast(self.beta_2, dtype)
        decay = ops.cast(0.96, dtype)
        schedule_decay = ops.cast(self.schedule_decay, dtype)
        mu_t = beta_1 * (1.0 - 0.5 * ops.power(decay, t * schedule_decay))
        mu_t1 = beta_1 * (1.0 - 0.5 * ops.power(decay, (t + 1.0) * schedule_decay))

        index = self._get_variable_index(variable)
        m, v, mu_product = self._momentums[index], self._velocities[index], self._mu_products[index]
        prod_t = mu_product * mu_t
        m_t = beta_1 * m + (1.0 - beta_1) * g
        v_t = beta_2 * v + (1.0 - beta_2) * ops.square(g)
        self.assign(mu_product, prod_t)
        self.assign(m, m_t)
        self.assign(v, v_t)

        g_prime = g / (1.0 - prod_t)
        m_prime = m_t / (1.0 - prod_t * mu_t1)
        v_prime = v_t / (1.0 - ops.power(beta_2, t))
        m_bar = (1.0 - mu_t) * g_prime + mu_t1 * m_prime
        self.assign_sub(variable, lr * m_bar / (ops.sqrt(v_prime) + self.epsilon))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "schedule_decay": self.schedule_decay,
            }
        )
        return config


def make_optimizer(cfg: TrainConfig) -> keras.optimizers.Optimizer:
    """``cfg.optimizer``: ``"nadam_legacy"`` (Keras-2 Nadam, default), ``"nadam"`` (Keras 3) or ``"adam"``."""
    common = {
        "learning_rate": float(cfg.learning_rate),
        "beta_1": float(cfg.beta_1),
        "beta_2": float(cfg.beta_2),
        "epsilon": float(cfg.epsilon),
    }
    if cfg.optimizer == "nadam_legacy":
        return LegacyNadam(schedule_decay=float(cfg.schedule_decay), **common)
    if cfg.optimizer == "nadam":
        return keras.optimizers.Nadam(**common)
    if cfg.optimizer == "adam":
        return keras.optimizers.Adam(**common)
    raise ValueError(f"Unknown optimizer {cfg.optimizer!r}; use 'nadam_legacy', 'nadam' or 'adam'")
