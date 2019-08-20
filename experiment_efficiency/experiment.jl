# Reactive message-passing in linear Gaussian dynamical system
# Experiment to pass messages without a schedule
#
# Wouter Kouw, BIASlab
# 12-08-2019

using Revise
using Distributions
using DataStructures
using LightGraphs
using MetaGraphs
using Plots
gr()

# Factor graph components
include("../nodes/node_gaussian.jl")
include("../nodes/transition_gaussian.jl")
include("../nodes/likelihood_gaussian.jl")
include("../edges/edge_gaussian.jl")
include("../edges/edge_delta.jl")
include("../util.jl")

# Data
include("../gen_data.jl")

"""
Experiment parameters
"""

# Signal time horizon
T = 100

# Reaction-time clock
TT = 20

# Spiking threshold
threshold = 1e-6

# Known transition and observation matrices
gain = 0.8
emission = 1.0

# Known noises (variance form)
process_noise = 2.0
measurement_noise = 0.5

# Parameters for state x_0
mean_0 = 0.0
precision_0 = 1.0

# Generate data
observed, hidden = gendata_LGDS(gain,
                                emission,
                                process_noise,
                                measurement_noise,
                                mean_0,
                                precision_0,
                                time_horizon=T)

"""
Model/graph specification

This assumes the following type of models
p(y_{1:T}, x_{0:T} | u_{1:T}) = p(x_0) Π_t p(y_t, x_t | x_{t-1})

In other words, Markov chains of time-slices of a state-space models.
Below, we specify the following model through the time-slice subgraph

    x_t-1    _       x_t
... --∘---->|_|----->|=|-----> ...
            g_t       |
                      |
                     |_| f_t
                      |
                      ∘
                     y_t

x_t-1 = previous state edge
g_t = state transition edge
x_t = current state edge
f_t = likelihood node
y_t = observation node
"""

# Start graph
graph = MetaGraph(PathDiGraph(5))

# Previous state
x_tmin = EdgeGaussian("x_tmin")
set_props!(graph, 1, Dict{Symbol,Any}(:object => :x_tmin, :id => "x_tmin"))

# State transition node
g_t = TransitionGaussian("g_t")
set_props!(graph, 2, Dict{Symbol,Any}(:object => :g_t, :id => "g_t"))

# Current state
x_t = EdgeGaussian("x_t")
set_props!(graph, 3, Dict{Symbol,Any}(:object => :x_t, :id => "x_t"))

# Observation likelihood node
f_t = LikelihoodGaussian("f_t")
set_props!(graph, 4, Dict{Symbol,Any}(:object => :f_t, :id => "f_t"))

# Observation
y_t = EdgeDelta("y_t")
set_props!(graph, 5, Dict{Symbol,Any}(:object => :y_t, :id => "y_t"))

# Ensure vertices can be recalled from given id
set_indexing_prop!(graph, :id)
set_indexing_prop!(graph, :object)

"""
Run inference procedure
"""

# Preallocation
num_ticks = zeros(T,)
estimated_states = zeros(T, 2)
free_energy_gradients = zeros(T,)

# Set state prior x_0
global x_t = EdgeGaussian("x_0"; mean=mean_0, precision=precision_0)

for t = 1:T

    # Report progress
    if mod(t, T/10) == 1
        println("At iteration "*string(t)*"/"*string(T))
    end

    # Previous state
    global x_tmin = EdgeGaussian("x_tmin", mean=x_t.mean, precision=x_t.precision)

    # State transition node
    global g_t = TransitionGaussian("g_t", edge_mean="x_tmin", edge_data="x_t", edge_precision=inv(process_noise), threshold=threshold)

    # Current state
    global x_t = EdgeGaussian("x_t", mean=0.0, precision=1.0)

    # Observation likelihood node
    global f_t = LikelihoodGaussian("f_t", edge_mean="x_t", edge_data="y_t", edge_precision=inv(measurement_noise), threshold=threshold)

    # Observation edge
    global y_t = EdgeDelta("y_t", observation=observed[t])

    # Start message routine
    act(x_tmin, belief(x_tmin), 1e12, graph)

    # Start clock for reactions
    tt = 0
    while ~g_t.silent & ~f_t.silent

        react(g_t, graph)
        # println(g_t.silent)
        react(x_t, graph)
        react(f_t, graph)
        println(f_t.silent)
        react(y_t, graph)
        println(f_t.silent)

        # Keep track of internal clock
        tt += 1
    end

    # Write out estimated state parameters
    estimated_states[t, 1] = x_t.mean
    estimated_states[t, 2] = sqrt(1/x_t.precision)

    # Keep track of FE gradients
    free_energy_gradients[t] = x_t.grad_free_energy

    # Store number of clock ticks
    num_ticks[t] = tt
end

"""
Visualize experimental results
"""

# Visualize final estimates over time
plot(hidden[2:end], color="red", label="states")
plot!(estimated_states[:,1], color="blue", label="estimates")
plot!(estimated_states[:,1],
      ribbon=[estimated_states[:,2], estimated_states[:,2]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
scatter!(observed, color="black", label="observations")
savefig(pwd()*"/experiment_efficiency/viz/state_estimates.png")

# Visualize free energy gradients over time-series
plot(free_energy_gradients, color="black", label="||dF||_t")
xlabel!("time (t)")
ylabel!("Norm of free energy gradient")
savefig(pwd()*"/experiment_efficiency/viz/FE_gradients.png")

# Visualize number of ticks of secondary clock
plot(num_ticks, color="black")
xlabel!("time (t)")
ylabel!("Number of ticks")
savefig(pwd()*"/experiment_efficiency/viz/num_ticks.png")
