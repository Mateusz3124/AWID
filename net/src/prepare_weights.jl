nfan(n) = 1, n
nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end])

function init(dims::Integer...; gain::Real=1)
    scale = Float32(gain) * sqrt(24.0f0 / sum(nfan(dims...)))
    (rand(Float32, dims...) .- 0.5f0) .* scale
end

function init_weight(in::Int64, out::Int64)
    return (Variable(init(out, in)))
end

function init_weight(height::Int64, width::Int64, depth::Int64)
    return (Variable(init(height, width, depth)))
end