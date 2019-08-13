# Schedule-free message-passing in linear Gaussian dynamical system
# Wouter Kouw, BIASlab
# 12-08-2019

using Revise
using Distributions
using DataStructures
using LighGraphs
using MetaGraphs
using Plots

# Factor graph components
include("../nodes/node_gaussian.jl")
include("../edges/edge_gaussian.jl")
include("../edges/edge_delta.jl")

# Data
include("gen_data.jl")

# Visualization
vizprf = pwd() * "/viz/"
viz = true

"""
Experiment parameters
"""

# Signal time horizon
T = 100

# Reaction-time clock
TT = 10

# Known transition and observation matrices
transition_matrix = 1.0
emission_matrix = 1.0

# Known noises
process_noise = 2.0
measurement_noise = 10.0

# Parameters for state x_0
m0 = 0.00
W0 = 0.01

# Generate data
observed, hidden = gen_data_randomwalk(process_noise,
                                       measurement_noise,
                                       m0, W0, time_horizon=T)

"""
Model/graph specification

This assumes the following type of models
p(y_{1:T}, x_{0:T} | u_{1:T}) = p(x_0) Π_t p(y_t, x_t | x_{t-1}, u_t)

In other words, Markov chains of time-slices of a state-space models.
"""

# Start graph
graph = MetaGraph(PathDiGraph(5))

# Previous state
global x_tmin = EdgeGaussian("x_tmin")
set_props!(graph, 1, Dict{Symbol,Any}(:object => :x_tmin, :type => "edge", :id => "x_tmin"))

# State transition node
global g_t = TransitionGaussian("g_t", edge_mean=:x_tmin, edge_data=:x_t, edge_precision=process_noise)
set_props!(graph, 2, Dict{Symbol,Any}(:object => :g_t, :type => "node", :id => "g_t")

# Current state
global x_t = EdgeGaussian("x_t")
set_props!(graph, 3, Dict{Symbol,Any}(:object => :x_t, :type => "edge", :id => "x_t")

# Observation likelihood node
global f_t = LikelihoodGaussian("f_t", edge_mean=:x_t, edge_data=:y_t, edge_precision=measurement_noise)
set_props!(graph, 4, Dict{Symbol,Any}(:object => :f_t, :type => "node", :id => "f_t")

# Observation
global y_t = EdgeDelta("y_t")
set_props!(graph, 5, Dict{Symbol,Any}(:object => :y_t, :type => "edge", :id => "y_t")

# Ensure vertices can be recalled from given id
set_indexing_prop!(graph, :id)

"""
Run inference procedure
"""

# Preallocation
estimated_states = zeros(T,2)

# Set state prior x_0
global x_t = EdgeGaussian("x_0"; mean=m0, precision=W0)

for t = 1:T

    # Report progress
    println("At iteration "*string(t)*"/"*string(T))

    # Previous state
    global x_tmin = EdgeGaussian("x_tmin", mean=x_t.params["mean"], x_t.params["precision"])

    # State transition node
    global g_t = TransitionGaussian("g_t", edge_mean=:x_tmin, edge_data=:x_t, edge_precision=process_noise)

    # Current state
    global x_t = EdgeGaussian("x_t", mean=x_tmin.params["mean"], x_tmin.params["precision"])

    # Observation likelihood node
    global f_t = LikelihoodGaussian("f_t", edge_mean=:x_t, edge_data=:y_t, edge_precision=measurement_noise)

    # Observation edge
    global y_t = EdgeDelta("y_t", observation=observed[t])

    # Start message routine
    act(x_tmin, message(z_t), 1e12, graph)

    # Start clock
    for tt = 1:TT
        react(f_t, graph)
        react(x_t, graph)
        react(g_t, graph)
        react(y_t, graph)
    end

    # Write out estimated state parameters
    estimated_states[t,1] = x_t.params["mean"]
    estimated_states[t,2] = x_t.params["precision"]
end

"""
Visualize experimental results
"""

# Visualize estimates
if viz
    plot(hidden[2:end], color="red", label="states")
    plot!(states_mean, color="blue", label="estimates")
    plot!(states_mean, ribbon=[100*1/sqrt.(states_prec), 100*1/sqrt.(states_prec)],
          linewidth = 2,
          color="blue",
          fillalpha = 0.2,
          fillcolor = "blue", label="")
    scatter!(observed, color="black", label="observations")
    savefig(pwd()*"/experiment-schedulefree/viz/state_estimates.png")
end
