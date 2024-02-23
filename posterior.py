#!/usr/bin/env python

import jax
import jax.numpy as jnp
import numpy as np

from mc import metropolis

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_debug_infs', True)

def cgc_sigma(s, Q2, x, Q2s0, gamma, ec):
    return 1

def cgc_error(s, Q2, x, Q2s0, gamma, ec):
    return 1


pr_mean = jnp.array([0.,1.,-2.]) 
pr_var = jnp.array([3.,1.,0.5])

def log_prior(Q2s0, gamma, ec):
    p0 = -1./2.*(Q2s0-pr_mean[0])**2/pr_var[0]**2 - jnp.log(pr_var[0]*jnp.sqrt(2*jnp.pi))
    p1 = -1./2.*(gamma-pr_mean[1])**2/pr_var[1]**2 - jnp.log(pr_var[1]*jnp.sqrt(2*jnp.pi))
    p2 = -1./2.*(ec-pr_mean[2])**2/pr_var[2]**2 - jnp.log(pr_var[2]*jnp.sqrt(2*jnp.pi)) 
    return p0 + p1 + p2

data = []

with open('data/redx-2009-parsed-small-x.txt', 'r') as f:
    lines = f.readlines()
    for i in range(1,len(lines)):
        r = [np.float64(x) for x in lines[i].split()]
        data.append(r)

data = np.array(data)

def log_likelihood(Q2s0, gamma, ec):
    ll = 0
    for i in range(5):
        s = data[i,0]
        Q2 = data[i,1]
        x = data[i,2]
        val_cgc = cgc_sigma(s, Q2, x, Q2s0, gamma, ec)
        var_cgc = cgc_error(s, Q2, x, Q2s0, gamma, ec)
        val_ex = data[i,3]
        var_ex = data[i,4] * data[i,3]
        ll -= (val_cgc - val_ex)**2 / (var_cgc**2 + var_ex**2)
    return ll

def log_posterior(params):
    Q2s0 = params[0]
    gamma = params[1]
    ec = params[2]
    lp = log_prior(Q2s0, gamma, ec)
    ll = log_likelihood(Q2s0, gamma, ec)
    return lp + ll

key = jax.random.PRNGKey(0)
chain = metropolis.Chain(log_posterior, jnp.zeros(3), key)
chain.step(N=1000)

Nskip = 100
Nsamples = 1000
for i in range(Nsamples):
    chain.step(N=Nskip)
    print(*chain.x)



