using Distributions

# Stochastic nodes
f_1(x, mu, sigma) = pdf(Normal(mu, sigma), x)
f_2(mu, u, s) = pdf(Normal(u, s), mu)

# Observation node
function f_0(x; x_hat=NaN)
    "
    Distribution for node f_0.

    Before observation, all probabilities are equal and set to 1.
    After observation, it only returns 1 if x is equal to the observed value.
    "
    # Check if observation has been made
    if isnan(x_hat)
        return 1
    else
        # If x == x_hat, probability = 1
        if x == x_hat
            return 1
        else
            return 0
        end
    end
end

function Hq_x(m_x, v_x)
    "Entropy of recognition distribution of edge x."
    return entropy(q_x(m_x, v_x))
end

function Hq_mu(m_mu, v_mu)
    "Entropy of recognition distribution of edge mu."
    return entropy(q_mu(m_mu, v_mu))
end

function Uf_1(m_x, v_x, m_mu, v_mu, sigma)
    "
    Internal energy of node f_1.

    Consists of the expected energy with respect to the relevant recognition
    distributions: E_qx E_qmu [ -log f_1(x, mu, sigma) ]

    Integration with respect to normals for mu and x leads to:
        - 1/2*(1/sigma^2)*(v_x + v_mu)
    "
    return -log(f_1(m_x, m_mu, sigma))
end

function Uf_2(m_mu, v_mu, u, s)
    "
    Internal energy of node f_2.

    Consists of the expected energy with respect to the relevant recognition
    distributions: E_qmu -log f_2(mu, u, s)

    Integration with respect to normals for mu and x leads to:
        - 1/2*(1/s^2)*v_mu
    "
    return -log(f_2(m_mu, u, s))
end

function Fq_x(m_x, v_x, m_mu, v_mu, sigma)
    "
    Edge-specific energy for edge of x

    It needs the following sets of parameters:
    - its own variational parameters
    - the variational parameters of all edges with recognition distributions
    belonging to all edges attached to the neighbouring nodes.
    - the clamped parameters of all neighbouring nodes.
    "
    # Entropy of edge x
    Hq = Hq_x(m_x, v_x)

    # Internal energy of node f_1
    U1q = Uf_1(m_x, v_x, m_mu, v_mu, sigma)

    # Combining terms
    return -Hq + 1/2*U1q
end

function Fq_mu(m_mu, v_mu, m_x, v_x, sigma, u, s)
    "
    Edge-specific energy for edge of x

    It needs the following sets of parameters:
    - its own variational parameters
    - the variational parameters of all edges with recognition distributions
    belonging to all edges attached to the neighbouring nodes.
    - the clamped parameters of all neighbouring nodes.
    "
    # Entropy of edge mu
    Hq = Hq_mu(m_mu, v_mu)

    # Internal energy of node f_1
    U1q = Uf_1(m_x, v_x, m_mu, v_mu, sigma)

    # Internal energy of node f_2
    U2q = Uf_2(m_mu, v_mu, u, s)

    # Combining terms
    return -Hq + 1/2*U1q + U2q
end

function DFq_x(m_x, v_x, m_x_min, v_x_min, m_mu, v_mu, sigma)
    "Delta free energy for new variational parameters."
    return Fq_x(m_x, v_x, m_mu, v_mu, sigma) - Fq_x(m_x_min, v_x_min, m_mu, v_mu, sigma)
end

function DFq_mu(m_mu, v_mu, m_mu_min, v_mu_min, m_x, v_x, sigma, u, t)
    "Delta free energy for new variational parameters."
    return Fq_mu(m_mu, v_mu, m_x, v_x, sigma, u, t) - Fq_mu(m_mu_min, v_mu_min, m_x, v_x, sigma, u, t)
end
