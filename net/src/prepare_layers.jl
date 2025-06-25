function graph_build(x::Variable, y::Variable, chain::Chain)
    if !(chain.elements[end] isa Classification)
        throw(ArgumentError("Last element of chain must be Classification"))
    end
    opt_values = Variable[]
    val = x
    for i in 1:(length(chain.elements)-1)
        el = chain.elements[i]
        val = build!(el, opt_values, val)
    end
    E = chain.elements[end].fun(val, y)
    return (topological_sort(E), val, opt_values)
end

function graph_build_from_opt_values(x::Variable, y::Variable, chain::Chain, opt_values::Vector{Variable})
    val = x
    iter = 1
    for i in 1:(length(chain.elements)-1)
        el = chain.elements[i]
        val, iter = build_from_opt_values!(el, opt_values, val, iter)
    end
    E = chain.elements[end].fun(val, y)
    return (topological_sort(E), val, opt_values)
end