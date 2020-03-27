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
T = 50

# Secondary clock for number of iterations
TT = 10

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
Random.seed!(256)
observed, hidden = gendata_LGDS(gain,
                                emission,
                                process_noise,
                                measurement_noise,
                                mean_0,
                                precision_0,
                                time_horizon=T)

"""
Model specification
"""

CPUtic()

g = FactorGraph()

@RV s_0 ~ GaussianMeanPrecision(mean_0, precision_0)

global s = Vector{Variable}(undef, T)
global x = Vector{Variable}(undef, T)
global s_t_min = s_0
for t = 1:T
    @RV s[t] ~ GaussianMeanPrecision(s_t_min, inv(process_noise))
    @RV x[t] ~ GaussianMeanVariance(s[t], inv(measurement_noise))

    global s_t_min = s[t]

    placeholder(x[t], :x, index=t)
end

"""
Mean-field algorithm generation
"""

# Define recognition factorization
RecognitionFactorization()

q_s_0 = RecognitionFactor(s_0)

q_s = Vector{RecognitionFactor}(undef, T)
for t=1:T
    q_s[t] = RecognitionFactor(s[t])
end

# Compile algorithm
algo_s_mf = variationalAlgorithm([q_s_0; q_s], name="SMF")
algo_F_mf = freeEnergyAlgorithm(name="MF");

"""
Run inference algorithm
"""

# Load algorithm
eval(Meta.parse(algo_s_mf))
eval(Meta.parse(algo_F_mf))

# Initialize data
global data = Dict(:x => observed)

# Initial recognition distributions
global marginals_mf = Dict{Symbol, ProbabilityDistribution}()
for t = 0:T
    global marginals_mf[:s_*t] = vague(GaussianMeanPrecision)
end

# Run algorithm
global estimated_states = zeros(T,2,TT)
global F_mf = Vector{Float64}(undef, TT)
for tt = 1:TT
    stepSMF!(data, marginals_mf)

    for t = 1:T
        estimated_states[t,1,tt] = mean(marginals_mf[:s_*t])
        estimated_states[t,2,tt] = sqrt(var(marginals_mf[:s_*t]))
    end

    # Free energy
    F_mf[tt] = freeEnergyMF(data, marginals_mf)
end

CPUtoc()

"""
Visualize experimental results
"""

# Visualize final estimates over time
scatter(1:T, observed, color="black", label="observations")
plot!(hidden[2:end], color="red", label="states")
plot!(estimated_states[:,1,end], color="blue", label="estimates")
plot!(estimated_states[:,1,end],
      ribbon=[estimated_states[:,2,end], estimated_states[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("time (t)")
savefig(joinpath(@__DIR__, "viz/baseline_state_estimates.png"))

# Visualize free energy
plot(1:TT, F_mf, color="green", label="")
xlabel!("#iterations")
ylabel!("free energy (F)")
savefig(joinpath(@__DIR__, "viz/baseline_fe-iters.png"))
