include("./prepare_weights.jl")

abstract type ChainElement end

struct Chain
    elements::Vector{ChainElement}
    Chain(args::ChainElement...) = new(collect(args))
end

struct Dense <: ChainElement
    in::Int64
    out::Int64
    fun::Union{typeof(relu),typeof(σ)}
end

struct Classification{F} <: ChainElement
    fun::F
end

struct Embedding <: ChainElement
    weight::Matrix{Float32}
end

struct Convolution{F} <: ChainElement
    kernel_height::Int64
    kernel_width::Int64
    kernel_depth::Int64
    fun::F
end

struct MaxPool <: ChainElement
    pool_size::Int64
end

struct Flatten <: ChainElement end

build!(el::Embedding, opt_values::Vector{Variable}, val::GraphNode) = begin
    w = Variable(el.weight)
    val = differential.embedding(val, w)
    push!(opt_values, w)
    return val
end

build!(el::Convolution, opt_values::Vector{Variable}, val::GraphNode) = begin
    w = init_weight(el.kernel_height, el.kernel_width, el.kernel_depth)
    res = differential.convolution(val, w)
    bias = Variable(zeros(Float32, 1, el.kernel_depth))
    val = el.fun(res .+ bias)
    push!(opt_values, w)
    push!(opt_values, bias)
    return val
end

build!(el::MaxPool, opt_values::Vector{Variable}, val::GraphNode) = begin
    if (size(val.output, 2) != el.pool_size)
        throw(ArgumentError("maxpool size doesn't match width"))
    end
    val = differential.maxpool(val, Variable(Int64[el.pool_size]))
    return val
end

build!(el::Flatten, opt_values::Vector{Variable}, val::GraphNode) = begin
    val = differential.flatten(val)
    return val
end

function dense(w, x, bias, activation)
    return activation(w * x .+ bias)
end

build!(el::Dense, opt_values::Vector{Variable}, val::GraphNode) = begin
    w = init_weight(el.in, el.out)
    bias = Variable(zeros(Float32, el.out))
    val = dense(w, val, bias, el.fun)
    push!(opt_values, w)
    push!(opt_values, bias)
    return val
end

build_from_opt_values!(::Embedding, opt_values::Vector{Variable}, val::GraphNode, iter::Int) = begin
    w = opt_values[iter]
    iter += 1
    val = differential.embedding(val, w)
    return val, iter
end

build_from_opt_values!(el::Convolution, opt_values::Vector{Variable}, val::GraphNode, iter::Int) = begin
    w = opt_values[iter]
    iter += 1
    res = differential.convolution(val, w)
    bias = opt_values[iter]
    iter += 1
    val = el.fun(res .+ bias)
    return val, iter
end

build_from_opt_values!(el::MaxPool, opt_values::Vector{Variable}, val::GraphNode, iter::Int) = begin
    val = differential.maxpool(val, Variable(Int64[el.pool_size]))
    return val, iter
end

build_from_opt_values!(::Flatten, opt_values::Vector{Variable}, val::GraphNode, iter::Int) = begin
    val = differential.flatten(val)
    return val, iter
end

build_from_opt_values!(el::Dense, opt_values::Vector{Variable}, val::GraphNode, iter::Int) = begin
    w = opt_values[iter]
    iter += 1
    bias = opt_values[iter]
    iter += 1
    val = dense(w, val, bias, el.fun)
    return val, iter
end
