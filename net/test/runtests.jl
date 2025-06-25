using Test
using Random
using differential
# using net

include("../src/prepare_weights.jl")
include("../src/chain.jl")
include("../src/prepare_layers.jl")
include("../src/adam.jl")
include("../src/prepare_data.jl")
include("../src/prepare_network.jl")

error_threshold = 0.000001f0

@testset "adam" begin
    expected = [
        0.001 0.001 0.001 0.001 0.001;
        0.001 0.001 0.001 0.001 0.001;
        0.001 0.001 0.001 0.001 0.001;
        0.001 0.001 0.001 0.001 0.001
    ]

    Random.seed!(123)
    x = Variable(rand(Float32, 4, 5))
    copy = deepcopy(x.output)
    Random.seed!(123)
    x.gradient .= rand(Float32, 4, 5)

    opt_values = Vector{Variable}([x])

    opt = Adam(opt_values)
    apply!(opt, opt_values)
    delta = copy .- x.output
    @test sum(delta .- expected) / length(delta) < error_threshold
end

@testset "input validation" begin
    X_train = rand(Float32, 12, 200)
    X_test = rand(Float32, 12, 200)
    y_train = rand(Float32, 200)
    y_test = rand(Float32, 200)
    chain = Chain(
        Dense(12, 1, σ),
    )
    @test_throws ArgumentError begin
        startNetwork(X_train, y_train, X_test, y_test, chain, 21, 1)
    end
    chain = Chain(
        Dense(12, 1, σ),
        Classification(binarycrossentropy)
    )
    @test_throws ArgumentError begin
        startNetwork(X_train, y_train, X_test, y_test, chain, 21, 1)
    end
    @test_throws ArgumentError begin
        startNetwork(rand(Float32, 12, 201), y_train, X_test, y_test, chain, 21, 1)
    end
    @test_throws ArgumentError begin
        startNetwork(X_test, y_train, rand(Float32, 12, 201), y_test, chain, 21, 1)
    end
    #checks if can run properly
    result = redirect_stdout(devnull) do
        startNetwork(X_train, y_train, X_test, y_test, chain, 20, 1)
    end
    @test isnothing(result)
end

@testset "graph" begin
    X_train = Variable(rand(Float32, 12, 200))
    y_train = Variable(rand(Float32, 200))
    chain = Chain(
        Dense(12, 1, σ),
        Classification(binarycrossentropy)
    )
    (graph, ŷ, opt_values) = graph_build(X_train, y_train, chain)
    @test length(graph) == 8
    @test graph[1] isa Variable
    @test graph[2] isa Variable
    @test typeof(graph[3]) == differential.BroadcastedOperator{typeof(LinearAlgebra.mul!),Matrix{Float32},Vector{Float32}}
    @test graph[4] isa Variable
    @test typeof(graph[5]) == differential.BroadcastedOperator{typeof(+),Matrix{Float32},Vector{Float32}}
    @test typeof(graph[6]) == differential.BroadcastedOperator{typeof(σ),Vector{Float32},Vector{Float32}}
    @test graph[7] isa Variable
    @test typeof(graph[8]) == differential.BroadcastedOperator{typeof(binarycrossentropy),Vector{Float32},Vector{Float32}}
    @test typeof(ŷ) == differential.BroadcastedOperator{typeof(σ),Vector{Float32},Vector{Float32}}
    @test length(opt_values) == 2
end

import Flux: glorot_uniform

@testset "weight init" begin
    Random.seed!(123)
    expected = glorot_uniform(3, 5, 3)
    Random.seed!(123)
    result = init_weight(3, 5, 3).output
    @test sum(result .- expected) / length(result) < error_threshold
end