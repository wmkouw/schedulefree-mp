function gen_data_randomwalk(P, R, T, m0, V0)
    "Draw a time-series according to a generative model"

    # Dimensionality
    d = size(P, 1)

    # Preallocate
    y = zeros(T, d)
    x = zeros(T+1, d)

    # Initialize state
    x[1, :] = randn(d).*sqrt(V0) .+ m0

    for t = 1:T

        # Evolve state
        x[t+1, :] = randn(d).*sqrt(P) + x[t, :]

        # Observe
        y[t, :] = randn(d).*sqrt(R) + x[t+1, :]

    end
    return y, x
end

function gen_data_kalmanf(A, B, P, R, m0, V0; time_horizon=100)
    "Draw a time-series according to a generative model"

    # Dimensionality
    d = size(A, 1)

    # Preallocate
    y = zeros(T, d)
    x = zeros(T+1, d)

    # Initialize state
    x[1, :] = randn(d).*sqrt(V0) .+ m0

    for t = 1:time_horizon

        # Evolve state
        x[t+1, :] = randn(d).*sqrt(P) + A*x[t, :]

        # Observe
        y[t, :] = randn(d).*sqrt(R) + B*x[t+1, :]

    end
    return y, x
end
