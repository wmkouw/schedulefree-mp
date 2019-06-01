using Test

include("../nodes/node_gaussian.jl")

@testset "NodeGaussian" begin

    A = 1.
    Q = 1.

    f = NodeGaussian("x0", "x1", A, Q, "f1")

    @test f.node_id == "f1"
    @test f.edge_data_id == "x0"
    @test f.edge_mean_id == "x1"
end