# This code was copied from https://gitlab.com/yyamauchi/thirring/mc/metropolis.py on 4/27/2023
from functools import partial

import jax
import jax.numpy as jnp

@jax.jit
def _split(k):
    return jax.random.split(k,3)

@partial(jax.jit, static_argnums=1)
def _normal(k, shape):
    return jax.random.normal(k, shape)

_uniform = jax.jit(jax.random.uniform)

class Chain:
    def __init__(self, logdist, x0, key, delta=0.2):
        self.logdist = logdist
        self.x = x0
        self.LD = self.logdist(self.x)
        self.delta = delta
        self._key = key
        self._recent = [False]

        def _accrej(kacc, x, ld, xp, ldp):
            lddiff = ldp - ld
            accepted = False
            acc = _uniform(kacc) < jnp.exp(lddiff)
            return jax.lax.cond(acc, lambda: (xp, ldp, True), lambda: (x, ld, False))

        self._accrej = jax.jit(_accrej)

    def step(self, N=1):
        self.LD = self.logdist(self.x)
        for _ in range(N):
            kstep, kacc, self._key = _split(self._key)
            xp = self.x + self.delta*_normal(kstep, self.x.shape)
            ldp = self.logdist(xp)
            self.x, self.LD, accepted = self._accrej(kacc, self.x, self.LD, xp, ldp)
            self._recent.append(accepted)
        self._recent = self._recent[-100:]

    def calibrate(self):
        # Adjust delta.
        while self.acceptance_rate() < 0.3 or self.acceptance_rate() > 0.5:
            if self.acceptance_rate() < 0.3:
                self.delta *= 0.98
            if self.acceptance_rate() > 0.5:
                self.delta *= 1.02
            self.step(N=100)

    def acceptance_rate(self):
        return sum(self._recent) / len(self._recent)


