module differential

include("./struct/struct.jl")
include("./operations/operators.jl")
include("./struct/struct_init.jl")
include("./operations/building.jl")
include("./operations/forward.jl")
include("./operations/backward.jl")

export relu, σ, binarycrossentropy, Variable, Operator, GraphNode, forward_pass!, backward_pass!, topological_sort,
    embedding, maxpool, flatten, convolution, forward!, backward!

end