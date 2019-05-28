# Script to run a small-scale experiment for a Kalman filter
# Wouter Kouw
# 24-05-2019
#
# Consider one time-slice of the model,
# with the following nodes:
#
# f(x_t, x_t-1) = N(x_t, Ax_t-1, tau^-1)
# g(y_t, x_t) = N(y_t | Bx_t, \gamma⁻¹)
#
# with
# q(x_t) = N(x | m_x, V_x)
# q(mu) = N(mu | m_mu, V_mu)
#
# We want the node f_1 to appropriately react to incoming messages.

using Revise
using Printf
using Distributions
using Plots

# Import functions
include("gen_data.jl")
include("free_energies.jl")
include("update_rules.jl")

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
y, x = gen_data_randomwalk(Q, R, T, m0, V0)

# Plot generated data
if viz  
    plot(x, label="states")
    scatter!(y, color="red", label="observations")
end

# Initialize recognition distribution
q(x) = Normal(0,1)

# Preallocate arrays
Fqx = zeros(1, T)
Fqmu = zeros(1, T)

# Loop over time-slices
for t = 1:T

    # Messages to edge x
    nu0 = nu_f0x(x_hat)
    nu1 = nu_f1x(m_mu[i-1], sigma)

    # Update q(x)
    m_x[i], V_x[i] = update_qx(m_x[i-1], V_x[i-1], nu0, nu1, observed=true)

    # Compute free energy of edge x
    Fqx[i] = Fq_x(m_x[i], V_x[i], m_mu[i-1], V_mu[i-1], sigma)
    DFqx[i] = DFq_x(m_x[i], V_x[i], m_x[i-1], V_x[i-1], m_mu[i-1], V_mu[i-1], sigma)

    # Messages to edge mu
    nu2 = nu_f1mu(m_x[i], sigma)
    nu3 = nu_f2mu(u, s)

    # Update q(mu)
    m_mu[i], V_mu[i] = update_qmu(m_mu[i-1], V_mu[i-1], nu2, nu3)

    # Compute free energy of edge mu
    Fqmu[i] = Fq_mu(m_mu[i], V_mu[i], m_x[i], V_x[i], sigma, u, s)
    DFqmu[i] = DFq_mu(m_mu[i], V_mu[i], m_mu[i-1], V_mu[i-1], m_x[i], V_x[i], sigma, u, s)

end
