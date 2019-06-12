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
m0 = 0.0
W0 = 0.1

# Generate data
observed, hidden = gen_data_randomwalk(Q, R, T, m0, W0)

# Preallocate estimated states
states_mean = zeros(T,)
states_prec = zeros(T,)

# Plot generated data
if viz
    plot(hidden, label="states")
    scatter!(observed, color="red", label="observations")
end

## Pass through FFG on a time-slice basis

# Initial state prior
global x_t = EdgeGaussian(m0, W0, "current", Dict{String, Symbol}(), "x_0")

for t = 1:T
    println("At iteration "*string(t)*"/"*string(T))

    # Previous state
    global z_t = EdgeGaussian(x_t.params["mean"],
                              x_t.params["precision"],
                              "previous",
                              Dict{String, Symbol}("right" => :f_t),
                              "z_"*string(t))

    # State transition node
    global f_t = NodeGaussian(:x_t, :z_t, A, Q, "f_"*string(t), verbose=true)

    # New state edge
    global x_t = EdgeGaussian(z_t.params["mean"],
                              z_t.params["precision"],
                              "current",
                              Dict{String, Symbol}("left" => :f_t, "bottom" => :g_t),
                              "x_"*string(t))

    # Observation node
    global g_t = NodeGaussian(:y_t, :x_t, H, R, "g_"*string(t), verbose=true)

    # Observation edge
    global y_t = EdgeDelta(observed[t],
                           Dict{String, Symbol}("top" => :g_t),
                           "y_"*string(t))

    # Start message routine
    react(z_t)

    # Start clock
    for tt = 1:10

        react(f_t)
        react(x_t)
        react(g_t)
        react(y_t)

        if (length(x_t.new_messages["data"])==0) & (length(x_t.new_messages["mean"])==0)
            println("Stopped clock at iteration "*string(tt)*" for t = "*string(t))
            break
        end
    end

    # Keep track of estimated states
    states_mean[t] = x_t.params["mean"]
    states_prec[t] = x_t.params["precision"]
end

# Visualize estimates
if viz
    plot(hidden, color="red", label="states")
    plot!(states_mean, color="blue", label="estimates")
    scatter!(observed, color="black", label="observations")
end
