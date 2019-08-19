# Reactive message-passing in linear Gaussian dynamical system.
# Experiment to infer transition coefficients.
#
# Wouter Kouw, BIASlab
# 15-08-2019

using Revise
using Distributions
using DataStructures
using LightGraphs, MetaGraphs
using Plots
gr()

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
TT = 20

# Known transition and observation matrices
gain = 1.0
emission = 1.0
gain_transition = 1.0

# Noise parameters (variance form)
measurement_noise = 0.5
process_noise = 1.0
gain_noise = 1.0

# Clamped parameters (mean-precision, shape-scale form)
x0_params = [0.0, 0.1]
a0_params = [0.0, 0.1]
Γx_params = [0.1, 10.0]
Γa_params = [0.1, 10.0]

# Generate data
observed, hidden = gendata_LGDS(gain,
                                emission,
                                process_noise,
                                measurement_noise,
                                x0_params[1],
                                x0_params[2],
                                time_horizon=T)

"""
Model/graph specification

This assumes the following type of models
p(y_{1:T}, x_{0:T} | u_{1:T}) = p(x0) Π_t p(y_t, x_t | x_{t-1})

In other words, Markov chains of time-slices of a state-space models.
Below, we specify the following model through the time-slice subgraph
           _
      Γ_a |_|
           |
       γ_a ∘
           |    a_t
..--∘---->|_|-->|=|-----> ...
  a_t-1   h_t    |     _
                 |    |_| Γ_x
                 |     |
                 |     ∘ γ_x
                 |_____|       x_t
... -----∘----->|_______|----->|=|-----> ...
        x_t-1      g_t          |
                                |
                               |_| f_t
                                |
                                ∘
                               y_t

x_t-1   = previous state edge
g_t     = state transition edge
γ_x     = process noise edge
Γ_x     = process noise prior node
a_t-1   = previous gain edge
h_t     = gain transition node
a_t     = current gain edge
γ_a     = gain noise edge
Γ_a     = gain noise prior node
x_t     = current state edge
f_t     = likelihood node
y_t     = observation node
"""

# Start graph
graph = MetaGraph(SimpleGraph(12))

# Previous state edge
x_tmin = EdgeGaussian("x_tmin")
set_props!(graph, 1, Dict{Symbol,Any}(:object => :x_tmin, :id => "x_tmin"))

# State transition node
g_t = TransitionGaussian("g_t")
set_props!(graph, 2, Dict{Symbol,Any}(:object => :g_t, :id => "g_t"))

# Process noise edge
γ_x = EdgeGamma("γ_x")
set_props!(graph, 3, Dict{Symbol,Any}(:object => :γ_x, :id => "γ_x"))

# Process noise prior node
Γ_x = NodeGamma("Γ_x")
set_props!(graph, 4, Dict{Symbol,Any}(:object => :Γ_x, :id => "Γ_x"))

# Previous gain edge
a_tmin = EdgeGaussian("a_tmin")
set_props!(graph, 5, Dict{Symbol,Any}(:object => :a_tmin, :id => "a_tmin"))

# Gain transition node
h_t = TransitionGaussian("h_t")
set_props!(graph, 6, Dict{Symbol,Any}(:object => :h_t, :id => "h_t"))

# Current gain edge
a_t = EdgeGaussian("a_t")
set_props!(graph, 7, Dict{Symbol,Any}(:object => :a_t, :id => "a_t"))

# Gain noise edge
γ_a = EdgeGamma("γ_a")
set_props!(graph, 8, Dict{Symbol,Any}(:object => :γ_a, :id => "γ_a"))

# Gain noise prior node
Γ_a = NodeGamma("Γ_a")
set_props!(graph, 9, Dict{Symbol,Any}(:object => :Γ_a, :id => "Γ_a"))

# Current state edge
x_t = EdgeGaussian("x_t")
set_props!(graph, 10, Dict{Symbol,Any}(:object => :x_t, :id => "x_t"))

# Observation likelihood node
f_t = LikelihoodGaussian("f_t")
set_props!(graph, 11, Dict{Symbol,Any}(:object => :f_t, :id => "f_t"))

# Observation edge
y_t = EdgeDelta("y_t")
set_props!(graph, 12, Dict{Symbol,Any}(:object => :y_t, :id => "y_t"))

add_edge!(graph, 1, 2) # x_t-1 -- g_t
add_edge!(graph, 2, 3) # g_t -- γ_x
add_edge!(graph, 3, 4) # γ_x -- Γ_x
add_edge!(graph, 2, 7) # g_t -- a_t
add_edge!(graph, 7, 6) # a_t -- h_t
add_edge!(graph, 6, 5) # h_t -- a_t-1
add_edge!(graph, 6, 8) # h_t -- γ_a
add_edge!(graph, 8, 9) # γ_a -- Γ_a
add_edge!(graph, 2, 10) # g_t -- x_t
add_edge!(graph, 10, 11) # x_t -- f_t
add_edge!(graph, 11, 12) # f_t -- y_t

# Ensure vertices can be recalled from given id
set_indexing_prop!(graph, :id)
set_indexing_prop!(graph, :object)

"""
Run inference procedure
"""

# Preallocation
estimated_states = zeros(T, 2, TT)
estimated_state_noise = zeros(T, 2, TT)
estimated_gains = zeros(T, 2, TT)
estimated_gain_noise = zeros(T, 2, TT)

# Set state prior x0
global x_t = EdgeGaussian("x0", mean=x0_params[1], precision=x0_params[2])
global a_t = EdgeGaussian("a0", mean=a0_params[1], precision=a0_params[2])

for t = 1:T

      # Report progress
      if mod(t, T/10) == 1
            println("At iteration "*string(t)*"/"*string(T))
      end

      # Previous state
      global x_tmin = EdgeGaussian("x_tmin", mean=x_t.mean, precision=x_t.precision, block=true)

      # State transition node
      global g_t = TransitionGaussian("g_t", edge_mean="x_tmin", edge_data="x_t", edge_precision="γ_x", edge_transition="a_t")

      # Process noise edge
      global γ_x = EdgeGamma("γ_x", shape=0.1, scale=10.0)

      # Process noise prior node
      global Γ_x = NodeGamma("Γ_x", edge_data="γ_x", edge_shape=Γx_params[1], edge_scale=Γx_params[2])

      # Previous gain edge
      global a_tmin = EdgeGaussian("a_tmin", mean=a_t.mean, precision=a_t.precision, block=true)

      # Gain transition node
      global h_t = TransitionGaussian("h_t", edge_data="a_t", edge_mean="a_tmin", edge_precision="γ_a", edge_transition=gain_transition)

      # Current gain edge
      global a_t = EdgeGaussian("a_t", mean=a_t.mean, precision=a_t.precision)

      # Gain noise edge
      global γ_a = EdgeGamma("γ_a", shape=0.1, scale=10.0)

      # Gain noise prior node
      global Γ_a = NodeGamma("Γ_a", edge_data="γ_a", edge_shape=Γa_params[1], edge_scale=Γa_params[2])

      # Current state
      global x_t = EdgeGaussian("x_t", mean=x_tmin.mean, precision=x_tmin.precision)

      # Observation likelihood node
      global f_t = LikelihoodGaussian("f_t", edge_mean="x_t", edge_data="y_t", edge_precision=inv(measurement_noise))

      # Observation edge
      global y_t = EdgeDelta("y_t", observation=observed[t])

      # Pass messages from previous time-slice
      act(x_tmin, belief(x_tmin), 1e12, graph);
      act(a_tmin, belief(a_tmin), 1e12, graph);

      # Start clock for reactions
      for tt = 1:TT

            react(g_t, graph)
            react(γ_x, graph)
            react(Γ_x, graph)
            react(h_t, graph)
            react(a_t, graph)
            react(γ_a, graph)
            react(Γ_a, graph)
            react(x_t, graph)
            react(f_t, graph)
            react(y_t, graph)

            # Write out estimated state parameters
            estimated_states[t, 1, tt] = x_t.mean
            estimated_states[t, 2, tt] = sqrt(1/x_t.precision)
            estimated_state_noise[t, 1, tt] = γ_x.shape * γ_x.scale
            estimated_state_noise[t, 2, tt] = sqrt(γ_x.shape * γ_x.scale^2)
            estimated_gains[t, 1, tt] = a_t.mean
            estimated_gains[t, 2, tt] = sqrt(1/a_t.precision)
            estimated_gain_noise[t, 1, tt] = γ_a.shape * γ_a.scale
            estimated_gain_noise[t, 2, tt] = sqrt(γ_a.shape * γ_a.scale^2)
      end
end

"""
Visualize experimental results
"""

# Visualize final state estimates over time
plot(hidden[2:end], color="red", label="states")
plot!(estimated_states[:,1,end], color="blue", label="estimates")
plot!(estimated_states[:,1,end],
      ribbon=[estimated_states[:,2,end], estimated_states[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
scatter!(observed, color="black", markersize=3, label="observations")
xlabel!("time (t)")
title!("State estimates, q(x_t)")
savefig(pwd()*"/experiment_hierarchical-transition/viz/state_estimates.png")

# Visualize state belief parameter trajectory
t = T
plot(estimated_states[t,1,1:end], color="blue", label="q(x_"*string(t)*")")
plot!(estimated_states[t,1,1:end],
      ribbon=[estimated_states[t,2,1:end], estimated_states[t,2,1:end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("iterations")
title!("Parameter trajectory of q(x_t) for t="*string(t))
savefig(pwd()*"/experiment_hierarchical-transition/viz/state_parameter_trajectory_t" * string(t) * ".png")

# Visualize final transition coefficient estimates over time
plot(gain*ones(T,1), color="black", label="transition")
plot!(estimated_gains[:,1,end], color="blue", label="estimates")
plot!(estimated_gains[:,1,end],
      ribbon=[estimated_gains[:,2,end], estimated_gains[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("time (t)")
title!("Transition coefficient estimates, q(a_t)")
savefig(pwd()*"/experiment_hierarchical-transition/viz/transition_estimates.png")

# Visualize transition coefficient belief parameter trajectory
t = T
plot(estimated_gains[t,1,1:end], color="blue", label="q(a_"*string(t)*")")
plot!(estimated_gains[t,1,1:end],
      ribbon=[estimated_gains[t,2,1:end], estimated_gains[t,2,1:end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("iterations")
title!("Parameter trajectory of q(a_t) for t="*string(t))
savefig(pwd()*"/experiment_hierarchical-transition/viz/transition_parameter_trajectory_t" * string(t) * ".png")

# Visualize final state noise estimates over time
plot(process_noise*ones(T,1), color="black", label="true process noise", linewidth=2)
plot!(estimated_state_noise[:,1,end], color="blue", label="estimates")
plot!(estimated_state_noise[:,1,end],
      ribbon=[estimated_state_noise[:,2,end], estimated_state_noise[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("time (t)")
title!("State noise estimates, q(γ_x)")
savefig(pwd()*"/experiment_hierarchical-transition/viz/state_noise_estimates.png")

# Visualize final gain noise estimates over time
plot(gain_noise*ones(T,1), color="black", label="true gain noise", linewidth=2)
plot!(estimated_gain_noise[:,1,end], color="blue", label="estimates")
plot!(estimated_gain_noise[:,1,end],
      ribbon=[estimated_gain_noise[:,2,end], estimated_gain_noise[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("time (t)")
title!("Gain noise estimates, q(γ_x)")
savefig(pwd()*"/experiment_hierarchical-transition/viz/gain_noise_estimates.png")
