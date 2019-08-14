# Schedule-free message-passing in linear Gaussian dynamical system
# Wouter Kouw, BIASlab
# 12-08-2019

using Revise
using Distributions
using DataStructures
using LightGraphs
using MetaGraphs
using Plots

# Factor graph components
include("../nodes/node_gamma.jl")
include("../nodes/node_gaussian.jl")
include("../nodes/transition_gaussian.jl")
include("../nodes/likelihood_gaussian.jl")
include("../edges/edge_gaussian.jl")
include("../edges/edge_gamma.jl")
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
TT = 5

# Known transition and observation matrices
transition_matrix = 1.0
emission_matrix = 1.0

# Noise parameters
measurement_noise = 0.5
process_noise_shape = 1.0
process_noise_rate = 3.0
process_noise = process_noise_shape/process_noise_rate

# Parameters for state x_0
mean_0 = 0.0
precision_0 = 1.0

# Generate data
observed, hidden = gendata_LGDS(transition_matrix,
                                emission_matrix,
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
             _
            |_| h_t
             |
             ∘ γ_t
    x_t-1    |       x_t
... --∘---->|_|----->|=|-----> ...
            g_t       |
                      |
                     |_| f_t
                      |
                      |
                      ∘
                     y_t

x_t-1 = previous state edge
g_t = state transition edge
γ_t = process noise edge
h_t = process noise prior node
x_t = current state edge
f_t = likelihood node
y_t = observation node
"""

# Start graph
graph = MetaGraph(SimpleGraph(7))

# Previous state edge
x_tmin = EdgeGaussian("x_tmin")
set_props!(graph, 1, Dict{Symbol,Any}(:object => :x_tmin, :id => "x_tmin"))

# State transition node
g_t = TransitionGaussian("g_t")
set_props!(graph, 2, Dict{Symbol,Any}(:object => :g_t, :id => "g_t"))

# Process noise edge
γ_t = EdgeGamma("γ_t")
set_props!(graph, 3, Dict{Symbol,Any}(:object => :γ_t, :id => "γ_t"))

# Process noise prior node
h_t = NodeGamma("h_t")
set_props!(graph, 4, Dict{Symbol,Any}(:object => :h_t, :id => "h_t"))

# Current state edge
x_t = EdgeGaussian("x_t")
set_props!(graph, 5, Dict{Symbol,Any}(:object => :x_t, :id => "x_t"))

# Observation likelihood node
f_t = LikelihoodGaussian("f_t")
set_props!(graph, 6, Dict{Symbol,Any}(:object => :f_t, :id => "f_t"))

# Observation edge
y_t = EdgeDelta("y_t")
set_props!(graph, 7, Dict{Symbol,Any}(:object => :y_t, :id => "y_t"))

add_edge!(graph, 1, 2)
add_edge!(graph, 2, 5)
add_edge!(graph, 3, 2)
add_edge!(graph, 3, 4)
add_edge!(graph, 5, 6)
add_edge!(graph, 6, 7)

# Ensure vertices can be recalled from given id
set_indexing_prop!(graph, :id)
set_indexing_prop!(graph, :object)

"""
Run inference procedure
"""

# Preallocation
estimated_states = zeros(T, 2, TT)
estimated_noises = zeros(T, 2, TT)

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
      global g_t = TransitionGaussian("g_t", edge_mean="x_tmin", edge_data="x_t", edge_precision="γ_t")

      # Process noise edge
      global γ_t = EdgeGamma("γ_t")

      # Process noise prior node
      global h_t = NodeGamma("h_t", edge_data="γ_t", edge_shape=process_noise_shape, edge_rate=process_noise_rate)

      # Current state
      global x_t = EdgeGaussian("x_t", mean=x_tmin.mean, precision=x_tmin.precision)

      # Observation likelihood node
      global f_t = LikelihoodGaussian("f_t", edge_mean="x_t", edge_data="y_t", edge_precision=inv(measurement_noise))

      # Observation edge
      global y_t = EdgeDelta("y_t", observation=observed[t])

      # Start message routine
      act(x_tmin, belief(x_tmin), 1e12, graph);

      # Start clock for reactions
      for tt = 1:TT

          react(g_t, graph)
          react(γ_t, graph)
          react(h_t, graph)
          react(x_t, graph)
          react(f_t, graph)
          react(y_t, graph)

          # Write out estimated state parameters
          estimated_states[t, 1, tt] = x_t.mean
          estimated_states[t, 2, tt] = x_t.precision
          estimated_noises[t, 1, tt] = γ_t.shape / γ_t.rate
          estimated_noises[t, 2, tt] = γ_t.shape / γ_t.rate^2
      end
end

"""
Visualize experimental results
"""

# Visualize final state estimates over time
plot(hidden[2:end], color="red", label="states")
plot!(estimated_states[:,1,end], color="blue", label="estimates")
plot!(estimated_states[:,1,end],
      ribbon=[1/estimated_states[:,2,end], 1/estimated_states[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
scatter!(observed, color="black", label="observations")
title!("Final state estimates")
savefig(pwd()*"/experiment_infer-processnoise/viz/state_estimates.png")

# Visualize final noise estimates over time
plot(estimated_noises[:,1,end], color="blue", label="estimates")
plot!(estimated_noises[:,1,end],
      ribbon=[1/estimated_noises[:,2,end], 1/estimated_noises[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
title!("Final noise estimates")
savefig(pwd()*"/experiment_infer-processnoise/viz/noise_estimates.png")

# Visualize state belief parameter trajectory
t = 100
plot(estimated_states[t,1,:], color="blue", label="q(x_"*string(t)*")")
plot!(estimated_states[t,1,:],
      ribbon=[1/estimated_states[t,2,:], 1/estimated_states[t,2,:]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
savefig(pwd()*"/experiment_infer-processnoise/viz/state_parameter_trajectory_t" * string(t) * ".png")

# Visualize noise belief parameter trajectory
t = 100
plot(estimated_noises[t,1,:], color="blue", label="q(γ_"*string(t)*")")
plot!(estimated_noises[t,1,:],
      ribbon=[1/estimated_noises[t,2,:], 1/estimated_noises[t,2,:]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
savefig(pwd()*"/experiment_infer-processnoise/viz/noise_parameter_trajectory_t" * string(t) * ".png")
