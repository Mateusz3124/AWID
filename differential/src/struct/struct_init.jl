import Base: eltype
import Base: *
import LinearAlgebra: mul!

Variable(output::T) where {T} = Variable{T}(output,
    zeros(eltype(T), size(output)))

ScalarOperator(fun, inputs...) = begin
    zeros1 = getInit(fun, (input.output for input in inputs)...)
    zeros2 = deepcopy(zeros1)
    ScalarOperator{typeof(fun),typeof(zeros1)}(inputs,
        zeros1, zeros2)
end

BroadcastedOperator(fun, inputs...; storage=Vector{Float32}(undef, 0)) = begin
    zeros1 = getInit(fun, (input.output for input in inputs)...)
    zeros2 = deepcopy(zeros1)
    BroadcastedOperator{typeof(fun),typeof(zeros1),typeof(storage)}(inputs,
        zeros1, zeros2, storage)
end

getparam(::Variable{T}) where {T} = T
getparam(::ScalarOperator{F,T}) where {F,T} = T
getparam(::BroadcastedOperator{F,T}) where {F,T} = T

const outputDict = Dict{Type,Type}(
    typeof(σ) => Vector{Float32},
    typeof(binarycrossentropy) => Vector{Float32},
    typeof(embedding) => Array{Float32,3},
    typeof(flatten) => Matrix{Float32},
    typeof(sum) => Vector{Float32}
)

function getInit(::typeof(mul!), x, y)
    if (size(y, 2)) == 1
        return zeros(Float32, size(x, 1))
    end
    return zeros(Float32, size(x, 1), size(y, 2))
end

function getInit(::typeof(*), x, y)
    return zeros(Float32, size(x))
end

function getInit(::typeof(/), x, y)
    return zeros(Float32, size(x))
end

function getInit(::typeof(+), x, y)
    return zeros(Float32, size(x))
end

function getInit(::typeof(-), x, y)
    return zeros(Float32, size(x))
end

function getInit(::typeof(sum), x)
    return Float32[0]
end

function getInit(::typeof(relu), x)
    return zeros(Float32, size(x))
end

function getInit(::typeof(σ), x)
    return zeros(Float32, size(x, 2))
end

function getInit(::typeof(binarycrossentropy), x, y)
    return Float32[0]
end

function getInit(::typeof(convolution), x, y)
    return zeros(Float32, size(x, 1) - size(y, 1) + 1, size(y, 3), size(x, 3))
end

function getInit(::typeof(embedding), x, y)
    return zeros(Float32, size(x, 1), size(y, 1), size(x, 2))
end

function getInit(::typeof(maxpool), x, y)
    return zeros(Float32, Int(size(x, 1) / y[1]), size(x, 2), size(x, 3))
end

function getInit(::typeof(flatten), x)
    return zeros(Float32, size(x, 1) * size(x, 2), size(x, 3))
end

import Base: show, summary
show(io::IO, x::ScalarOperator{F,T}) where {F,T} = print(io, "\n ┣━ op (", F, " ", T, ")");
show(io::IO, x::BroadcastedOperator{F,T}) where {F,T} = print(io, "\n ┣━ op. (", F, " ", T, ")");
show(io::IO, x::Variable{T}) where {T} = begin
    print(io, "\n ┣━ var (", T, ")")
    print(io, "\n ┣━ ^ ")
    summary(io, x.output)
    print(io, "\n ┗━ ∇ ")
    summary(io, x.gradient)
end