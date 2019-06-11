# Script to run a small-scale experiment for a Kalman filter
# Wouter Kouw
# 24-05-2019
#
# Consider one time-slice of the model,
# with the following nodes:
#
# f(x_t, x_t-1) = N(x_t | A x_t-1, Q)
# g(y_t, x_t) = N(y_t | H x_t, R)
#
# with
# q(x_t) = N(mt, Vt)
#
# Nodes should only react to incoming messages.

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
W0 = 1.

# Generate data
observed, hidden = gen_data_randomwalk(Q, R, T, m0, W0)

# Plot generated data
if viz  
    plot(hidden, label="states")
    scatter!(observed, color="red", label="observations")
end

## Pass through FFG on a time-slice basis

# Initial state prior
x_t = EdgeGaussian(mean=m0, 
                   precision=W0,
                   state_type="current",
                   nodes=Dict{String, Symbol}(),
                   id="x_0")

for t = 1:T

    # Previous state
    z_t = EdgeGaussian(mean=x_t.params["mean"], 
                       precision=x_t.params["precision"], 
                       state_type="previous",
                       nodes=Dict{String, Symbol}("right" => :f_t),
                       id="z_"*string(t))

    # State transition node
    f_t = NodeGaussian(edge_data_id=:x_t,
                       edge_mean_id=:z_t,
                       transition=A,
                       precision=Q,
                       id="f_"*string(t))

    # New state edge
    x_t = EdgeGaussian(mean=z_t.params["mean"], 
                       precision=z_t.params["precision"], 
                       state_type="current",
                       nodes=Dict{String, Symbol}("left" => :f_t, "bottom" => :gt),
                       id="x_"*string(t))

    # Observation node
    g_t = NodeGaussian(edge_data_id=:y_t,
                       edge_mean_id=:x_t,
                       transition=H,
                       precision=R,
                       id="g_"*string(t))

    # Observation edge
    y_t = EdgeDelta(data=observed[t],
                    node=:g_t,
                    id="y_"*string(t))

    # Start clock
    for tt = 1:10

        react(z_t)
        react(f_t)
        react(x_t)
        react(g_t)
        react(y_t)

    end
end