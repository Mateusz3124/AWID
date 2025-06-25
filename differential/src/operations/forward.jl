import Base: eltype

reset!(node::Variable) = begin
    fill!(node.gradient, 0.0f0)
end
reset!(node::Operator) = begin
    fill!(node.gradient, 0.0f0)
end

compute!(node::Variable) = nothing
compute!(node::Operator) =
    forward!(node, [input.output for input in node.inputs]...)

function forward_pass!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end