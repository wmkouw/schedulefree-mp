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
V0 = 1.

# Generate data
observed, hidden = gen_data_randomwalk(Q, R, T, m0, V0)

# Plot generated data
if viz  
    plot(hidden, label="states")
    scatter!(observed, color="red", label="observations")
end

## Pass through FFG on a time-slice basis

# Initialize variables
z_t = EdgeGaussian()
f_t = NodeGaussian()
x_t = EdgeGaussian()
g_t = NodeGaussian()
y_t = EdgeDelta()

# Initial state (t=0)
x_t = EdgeGaussian(mean=m0, 
                   precision=V0, 
                   node_l=Nothing, 
                   node_r=:f_t,
                   node_b=Nothing,
                   id="x0")

for t = 1:T

    # Edge x_t becomes edge x_t-1 (= z_t)
    z_t = x_t 

    # Transition node
    f_t = NodeGaussian(edge_data_id=:x_t,
                       edge_mean_id=:z_t,
                       transition=A,
                       precision=Q,
                       id="f"*string(t))

    # New state edge
    x_t = EdgeGaussian(mean=z_t.params["mean"], 
                       precision=z_t.params["precision"], 
                       node_l=:f_t, 
                       node_r=:f_t,
                       node_b=:g_t,
                       id="x"*string(t))

    # Observation node
    g_t = NodeGaussian(edge_data_id=:y_t,
                       edge_mean_id=:x_t,
                       transition=H,
                       precision=R,
                       id="g"*string(t))

    # Data point edge
    y_t = EdgeDelta(data=observed[t],
                    node=:g_t,
                    id="y"*string(t))

    # Start clock
    for tt = 1:10

        react(z_t)
        react(f_t)
        react(x_t)
        react(g_t)
        react(y_t)

    end
end