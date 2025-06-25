function visit(node::GraphNode, visited::Set{GraphNode}, order::Vector{GraphNode})
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end
    
function visit(node::Operator, visited::Set{GraphNode}, order::Vector{GraphNode})
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set{GraphNode}()
    order = GraphNode[]
    visit(head, visited, order)
    return order
end