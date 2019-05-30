# Script to run a small-scale experiment for a Kalman filter
# Wouter Kouw
# 24-05-2019
#
# Consider one time-slice of the model,
# with the following nodes:
#
# f(x_t, x_t-1) = N(x_t, Ax_t-1, tau^-1)
# g(y_t, x_t) = N(y_t | Bx_t, \gamma⁻¹)
#
# with
# q(x_t) = N(x | m_x, V_x)
# q(mu) = N(mu | m_mu, V_mu)
#
# We want the node f_1 to appropriately react to incoming messages.

using Revise
using Printf
using Distributions
using Plots

import nodes: node_gaussian
import edges: edge_gaussian, edge_delta

# Import functions
include("gen_data.jl")

# Prefixes for saving files
vizprf = pwd() * "/viz/"

# Visualize
viz = false

# Time horizon
T = 100

# Known transition and observation matrices
A = 1.
H = 1.

# Known noises
Q = 1.
R = 1.

# Parameters for state prior
m0 = 0.
V0 = 1.

# Generate data
y, x = gen_data_randomwalk(Q, R, T, m0, V0)

# Plot generated data
if viz  
    plot(x, label="states")
    scatter!(y, color="red", label="observations")
end

## Construct a time-slice of an FFG

# Initial state
x0 = edge(observed=false)

for t = 1:T

    # Previous state edge
    xt_min = xt

    # Transition node
    ft = node_gaussian(mean=A*xt_1, variance=Q, xt)

    # New state edge
    xt = edge_gaussian(ft-1, ft, observed=false)

    # Observation node
    gt = node_gaussian(mean=H*xt, variance=R, yt)

    # Data point edge
    yt = edge_delta(y[t])

    # Pass messages around
    pass()

end