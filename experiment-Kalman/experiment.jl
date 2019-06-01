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

# Factor graph components
include("../nodes/node_gaussian.jl")
include("../edges/edge_gaussian.jl")
include("../edges/edge_delta.jl")

# Data
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
observed, hidden = gen_data_randomwalk(Q, R, T, m0, V0)

# Plot generated data
if viz  
    plot(x, label="states")
    scatter!(y, color="red", label="observations")
end

## Pass through FFG on a time-slice basis

# Initial state
x_t = EdgeGaussian(mean=m0, 
                   precision=V0, 
                   id="x0", 
                   node_l=Nothing, 
                   node_r="f1",
                   node_b=Nothing)

for t = 1:T

    """Construct FFG time-slice"""

    # Edge x_t becomes edge x_t-1
    x_tmin1 = x_t 

    # Transition node
    ft = NodeGaussian(rvar=x_t,
                      mean=A*x_tmin1.params["mean"], 
                      precision=Q,
                      id="f"*string(t),
                      edge0="x"*string(t-1),
                      edge1="x"*string(t))

    # New state edge
    xt = EdgeGaussian(mean=x_tmin1.params["mean"], 
                      precision=x_tmin1.params["precision"],
                      id="e"*string(t), 
                      node_l="f"*string(t), 
                      node_r="f"*string(t+1),
                      node_b="g"*string(t))

    # Observation node
    gt = NodeGaussian(rvar=y_t,
                      mean=H*xt, 
                      precision=R,
                      id="g"*string(t),
                      edge0="x"*string(t-1),
                      edge1="y"*string(t))

    # Data point edge
    y_t = EdgeDelta(data=observed[t],
                    id="y"*string(t),
                    node="g"*string(t))

    # Start clock
    while 1

        react(x_tmin1)
        react(ft)
        react(x_t)
        react(gt)
        react(y_t)

    end
end