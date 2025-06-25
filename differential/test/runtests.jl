using Test
using differential
using Zygote
using Random
using NNlib

Random.seed!(0)

# sum is used in zygote because its gradient is all ones
# due to difference between implementations, results can differ
error_threshold = 0.000001f0

@testset "basic gradients" begin
    variable_a = Variable([2.0f0])
    variable_b = Variable([5.0f0])
    variable_c = Variable([-2.0f0])
    graph = topological_sort(variable_a .* variable_b)
    @test forward_pass!(graph)[1] == 10.0f0
    backward_pass!(graph)
    @test variable_a.gradient[1] == 5.0f0
    @test variable_b.gradient[1] == 2.0f0
    graph = topological_sort(variable_a .+ variable_b)
    @test forward_pass!(graph)[1] == 7.0f0
    backward_pass!(graph)
    @test variable_a.gradient[1] == 1.0f0
    @test variable_b.gradient[1] == 1.0f0

    graph = topological_sort(differential.relu(variable_a))
    @test forward_pass!(graph)[1] == 2.0f0
    backward_pass!(graph)
    @test variable_a.gradient[1] == 1.0f0
    graph = topological_sort(differential.relu(variable_c))
    @test forward_pass!(graph)[1] == 0.0f0
    backward_pass!(graph)
    @test variable_c.gradient[1] == 0.0f0

    variable_d = Variable([2.0f0 1.0f0])
    graph = topological_sort(differential.σ(variable_d))
    @test forward_pass!(graph) == [0.880797f0, 0.7310586f0]
    graph[2].gradient .= [1.0f0, 1.0f0]
    backward!(graph[2], variable_d)
    @test variable_d.gradient == [0.10499363f0 0.19661193f0]

    variable_e = Variable(rand(Float32, 10, 2, 5))
    graph = topological_sort(flatten(variable_e))
    x = variable_e.output
    @test forward_pass!(graph) == reshape(x, :, size(x)[end])
    last(graph).gradient .= ones(Float32, 20, 5)
    backward!(last(graph), variable_e)
    expected = Zygote.gradient((val) -> sum(reshape(val, :, size(val)[end])), x)
    @test sum(variable_e.gradient .- expected[1]) == 0.0f0
end

const ϵ = Float32(1e-8)

using Statistics: mean

#taken from flux website
function binarycrossentropy(ŷ, y; agg=mean)
    agg(@.(-y * log(ŷ + ϵ) - (1 - y) * log(1 - ŷ + ϵ)))
end

@testset "complex gradients" begin
    data = Variable(rand(Float32, 130, 50, 2))
    weight = Variable(rand(Float32, 3, 50, 8))
    graph = topological_sort(convolution(data, weight))
    @test sum(forward_pass!(graph) .- NNlib.conv(data.output, weight.output)) < error_threshold
    last(graph).gradient .= ones(Float32, 128, 8, 2)
    backward!(last(graph), data, weight)
    expected = Zygote.gradient((y, w) -> sum(NNlib.conv(y, w)), data.output, weight.output)
    @test sum(data.gradient .- expected[1]) / length(data.output) < error_threshold
    @test sum(weight.gradient .- expected[2]) / length(weight.output) < error_threshold

    data = Variable(rand(Float32, 128, 8, 2))
    graph = topological_sort(differential.maxpool(data, Variable([8])))
    @test sum(forward_pass!(graph) .- NNlib.maxpool(data.output, (8,))) / length(data.output) < error_threshold
    last(graph).gradient .= ones(Float32, 16, 8, 2)
    backward!(last(graph), data, Variable([8]))
    expected = Zygote.gradient((x) -> sum(NNlib.maxpool(x, (8,))), data.output)
    @test sum(data.gradient .- expected[1]) / length(data.output) < error_threshold

    ŷ = Variable(rand(Float32, 100))
    y = Variable(rand(Float32, 100))
    graph = topological_sort(differential.binarycrossentropy(ŷ, y))
    @test forward_pass!(graph)[1] == binarycrossentropy(ŷ.output, y.output)
    backward_pass!(graph)
    expected = Zygote.gradient((x, y) -> binarycrossentropy(x, y), ŷ.output, y.output)
    @test sum(ŷ.gradient .- expected[1]) / length(ŷ.output) < error_threshold
    @test sum(y.gradient .- expected[2]) / length(y.output) < error_threshold

    x = Variable(Int32.(rand(1:10, 10, 2)))
    w = Variable(rand(Float32, 10, 10))
    graph = topological_sort(embedding(x, w))
    expected_forward = permutedims(reshape(NNlib.gather(w.output, vec(x.output)), :, size(x.output)...), (2, 1, 3))
    @test forward_pass!(graph) == expected_forward
    grad = ones(Float32, size(expected_forward)...)
    last(graph).gradient .= grad
    result = backward!(last(graph), x, w)
    expected_grad = Zygote.gradient((x_, w_) -> sum(permutedims(reshape(NNlib.gather(w_, vec(x_)), :, size(x_)...), (2, 1, 3))), x.output, w.output)
    @test sum(w.gradient .- expected_grad[2]) / length(w.output) < error_threshold
end