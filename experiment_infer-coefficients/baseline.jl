# Reactive message-passing in linear Gaussian dynamical system
# Experiment to pass messages without a schedule
#
# Wouter Kouw, BIASlab
# 12-08-2019

using Random
using Revise
using CPUTime
using ForneyLab
using Plots
pyplot()

# Functions for generating data
include(joinpath(@__DIR__, "../gen_data.jl"))

"""
Experiment parameters
"""

# Signal time horizon
T = 30

# Reaction-time clock
TT = 10

# Known transition and observation matrices
transition_coefficient = 0.6
likelihood_coefficient = 1.0

# Noise parameters
process_noise = 0.5
measurement_noise = 0.3

# Clamped parameters
x0 = [0.1, 1.0]
a0 = [1.0, 1.0]

# Generate data
observed, hidden = gendata_LGDS(transition_coefficient,
                                likelihood_coefficient,
                                process_noise,
                                measurement_noise,
                                x0[1],
                                x0[2],
                                time_horizon=T)

"""
Model specification
"""

# Define composite transition node
@composite TransitionGaussian (x_t, a, x_tmin) begin
    @RV x_tmin_ = x_tmin*a
    @RV x_t ~ GaussianMeanVariance(x_tmin_, process_noise)
end

@NaiveVariationalRule(:node_type     => TransitionGaussian,
                      :outbound_type => Message{GaussianMeanPrecision},
                      :inbound_types => (Nothing, Message{Gaussian}, Message{Gaussian}))


# Start timer
CPUtic()

g = FactorGraph()

@RV a ~ GaussianMeanVariance(a0[1], a0[2])
@RV x_0 ~ GaussianMeanVariance(x0[1], x0[2])

global x = Vector{Variable}(undef, T)
global y = Vector{Variable}(undef, T)
global x_tmin = x_0
for t = 1:T

    # @RV x_tmin_ = a*x_tmin
    # @RV x[t] ~ GaussianMeanVariance(x_tmin_, process_noise)

    @RV x[t] ~ TransitionGaussian(x_tmin, a)
    @RV y[t] ~ GaussianMeanVariance(x[t], measurement_noise)

    global x_tmin = x[t]

    placeholder(y[t], :y, index=t)
end

"""
Mean-field algorithm generation
"""

# Define recognition factorization
RecognitionFactorization()

q_a = RecognitionFactor(a)
q_x_0 = RecognitionFactor(x_0)

q_x = Vector{RecognitionFactor}(undef, T)
for t=1:T
    q_x[t] = RecognitionFactor(x[t])
end

# Compile algorithm
algo_a_mf = variationalAlgorithm(q_a, name="AMF")
algo_x_mf = variationalAlgorithm([q_x_0; q_x], name="SMF")
algo_F_mf = freeEnergyAlgorithm(name="MF");

"""
Run inference algorithm
"""

# Load algorithm
eval(Meta.parse(algo_a_mf))
eval(Meta.parse(algo_x_mf))
eval(Meta.parse(algo_F_mf))

# Initialize data
global data = Dict(:y => observed)

# Initial recognition distributions
global marginals_mf = Dict{Symbol, ProbabilityDistribution}()
global marginals_mf[:a] = vague(GaussianMeanPrecision)
for t = 0:T
    global marginals_mf[:s_*t] = vague(GaussianMeanPrecision)
end

# Run algorithm
global estimated_states = zeros(T,2,TT)
global F_mf = Vector{Float64}(undef, TT)
for tt = 1:TT
    stepAMF!(data, marginals_mf)
    stepSMF!(data, marginals_mf)

    for t = 1:T
        estimated_states[t,1,tt] = mean(marginals_mf[:x_*t])
        estimated_states[t,2,tt] = sqrt(var(marginals_mf[:x_*t]))
    end

    global F_mf[tt] = freeEnergyMF(data, marginals_mf)
end

CPUtoc()

"""
Visualize experimental results
"""

# Visualize final estimates over time
plot(hidden[2:end], color="red", label="states")
plot!(estimated_states[:,1,end], color="blue", label="estimates")
plot!(estimated_states[:,1,end],
      ribbon=[estimated_states[:,2,end], estimated_states[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
scatter!(observed, color="black", label="observations")
savefig(joinpath(@__DIR__, "viz/state_estimates_baseline.png"))
