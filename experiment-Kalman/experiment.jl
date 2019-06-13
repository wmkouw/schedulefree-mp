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
using Distributions
using DataStructures
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
viz = true

# Time horizon
T = 100

# Known transition and observation matrices
transition_matrix = 0.8
emission_matrix = 1.0

# Known noises
transition_precision = 1.0
emission_precision = 0.5

# Parameters for state prior
m0 = 0.00
W0 = 0.01

# Generate data
observed, hidden = gen_data_kalmanf(transition_matrix,
                                    emission_matrix,
                                    transition_precision,
                                    emission_precision,
                                    m0, W0,
                                    time_horizon=T)

## Pass through FFG on a time-slice basis

# Preallocation
states_mean = zeros(T,)
states_prec = zeros(T,)
num_passes = zeros(T,)

# Initial state prior
global x_t = EdgeGaussian(m0, W0, "current", Dict{String, Symbol}("right" => :f_t), "x_0")

for t = 1:T
# t=1
    println("At iteration "*string(t)*"/"*string(T))

    # Previous state
    global z_t = EdgeGaussian(x_t.params["mean"],
                              x_t.params["precision"],
                              "previous",
                              Dict{String, Symbol}("right" => :f_t),
                              "z_"*string(t),
                              free_energy=x_t.free_energy)

    # State transition node
    global f_t = NodeGaussian(:x_t, :z_t,
                              transition_matrix,
                              transition_precision,
                              "f_"*string(t),
                              threshold=1e-6,
                              verbose=true)

    # New state edge
    global x_t = EdgeGaussian(z_t.params["mean"],
                              z_t.params["precision"],
                              "current",
                              Dict{String, Symbol}("left" => :f_t, "bottom" => :g_t),
                              "x_"*string(t))

    # Observation node
    global g_t = NodeGaussian(:y_t, :x_t,
                              emission_matrix,
                              emission_precision,
                              "g_"*string(t),
                              threshold=1e-6,
                              verbose=true)

    # Observation edge
    global y_t = EdgeDelta(observed[t],
                           Dict{String, Symbol}("top" => :g_t),
                           "y_"*string(t))

    # Start message routine
    act(z_t, message(z_t), 1e12)

    # Start clock
    for tt = 1:10
        react(f_t)
        react(x_t)
        react(g_t)
        react(y_t)

        if (tt > 3) & (y_t.prediction_error < 1e-3)
            println("Stopped clock at iteration "*string(tt)*" for t = "*string(t))
            num_passes[t] = tt
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
    plot!(states_mean, ribbon=[1/sqrt.(states_prec), 1/sqrt.(states_prec)],
      linewidth = 2,
      color="blue",
      fillalpha = 0.2,
      fillcolor = "blue", label="")
    scatter!(observed, color="black", label="observations")
    savefig(pwd()*"/experiment-Kalman/viz/state_estimates.png")

    # plot(num_passes, label="Number of clock ticks")
end
