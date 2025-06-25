abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Variable{T} <: GraphNode
    output::T
    gradient::T
end

struct ScalarOperator{F,T<:Union{Float32, AbstractArray{Float32}}} <: Operator
    inputs::Tuple{Vararg{GraphNode}}
    output::T
    gradient::T
end

struct BroadcastedOperator{F, T<:Union{Float32, AbstractArray{Float32}}, B} <: Operator
    inputs::Tuple{Vararg{GraphNode}}
    output::T
    gradient::T
    storage::B
end