module net

using differential
export relu, σ, Variable, Operator, forward_pass!, backward_pass!, topological_sort, binarycrossentropy

include("./chain.jl")
include("./adam.jl")
include("./prepare_layers.jl")
include("./prepare_data.jl")
include("./prepare_network.jl")

export startNetwork, Chain, Dense, Classification, Embedding, Convolution, Flatten, MaxPool, init_data, graph_build, graph_build_from_opt_values, accuracy, Adam, apply!
end