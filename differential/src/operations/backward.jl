function backward_pass!(order::Vector; seed=Float32[1.0])
    result = last(order)
    result.gradient .= seed
    for node in reverse(order)
        backward_pass!(node)
    end
end

function backward_pass!(node::Variable) end
function backward_pass!(node::Operator)
    backward!(node, node.inputs...)
end