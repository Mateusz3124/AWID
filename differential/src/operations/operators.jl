import Base: *
import LinearAlgebra: mul!
using Base.Threads

const ϵ = Float32(1e-8)

*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward!(node::BroadcastedOperator{typeof(mul!)}, A, x) = begin
    node.output .= A * x
end
backward!(node::BroadcastedOperator{typeof(mul!)}, A::GraphNode, x::GraphNode) = begin
    A.gradient .= node.gradient * x.output'
    x.gradient .= A.output' * node.gradient
end

import LinearAlgebra: diagm
Base.Broadcast.broadcasted(::typeof(*), x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward!(node::BroadcastedOperator{typeof(*)}, x, y) = begin
    node.output .= x .* y
end
backward!(node::BroadcastedOperator{typeof(*)}, x::GraphNode, y::GraphNode) = begin
    x.gradient .= y.output .* node.gradient
    y.gradient .= x.output .* node.gradient
end

Base.Broadcast.broadcasted(::typeof(/), x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward!(node::BroadcastedOperator{typeof(/)}, x, y) = begin
    node.output .= x ./ y
end
backward!(node::BroadcastedOperator{typeof(/)}, x::GraphNode, y::GraphNode) = begin
    x.gradient .= node.gradient ./ y.output
    y.gradient .= -node.gradient .* x.output ./ (y.output .^ 2)
end

Base.Broadcast.broadcasted(::typeof(+), x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward!(node::BroadcastedOperator{typeof(+)}, x, y) = begin
    node.output .= x .+ y
end
backward!(node::BroadcastedOperator{typeof(+)}, x::GraphNode, y::GraphNode) = begin
    x.gradient .= node.gradient
    if y.output isa Matrix{Float32}
        y.gradient .= dropdims(sum(node.gradient, dims=(1, 3)), dims=3)
    elseif y.output isa Vector{Float32}
        y.gradient .= vec(sum(node.gradient, dims=2))
    else
        y.gradient .= node.gradient
    end
end

Base.Broadcast.broadcasted(::typeof(-), x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward!(node::BroadcastedOperator{typeof(-)}, x, y) = begin
    node.output .= x .- y
end
backward!(node::BroadcastedOperator{typeof(-)}, x::GraphNode, y::GraphNode) = begin
    x.gradient .= node.gradient
    if y.output isa Matrix{Float32}
        y.gradient .= dropdims(sum(-node.gradient, dims=(1, 3)), dims=3)
    elseif y.output isa Vector{Float32}
        y.gradient .= vec(sum(-node.gradient, dims=2))
    else
        y.gradient .= -node.gradient
    end
end

import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x::GraphNode)
forward!(node::BroadcastedOperator{typeof(sum)}, x) = begin
    node.output .= sum(x)
end
backward!(node::BroadcastedOperator{typeof(sum)}, x::GraphNode) = begin
    𝟏 = ones(Float32, length(x.output))
    x.gradient .= 𝟏 .* node.gradient[1]
end

relu(x::GraphNode) = BroadcastedOperator(relu, x::GraphNode)
forward!(node::BroadcastedOperator{typeof(relu)}, x) = begin
    node.output .= max.(zero(x), x)
end
backward!(node::BroadcastedOperator{typeof(relu)}, x::GraphNode) = begin
    x.gradient .= node.gradient .* (x.output .> 0.0f0)
end

σ(x) = BroadcastedOperator(σ, x::GraphNode)
forward!(node::BroadcastedOperator{typeof(σ)}, x) = begin
    node.output .= vec(1.0f0 ./ (1.0f0 .+ exp.(-x)))
end
backward!(node::BroadcastedOperator{typeof(σ)}, x::GraphNode) = begin
    y = node.output
    one_vec = ones(Float32, length(y))
    x.gradient .= (y .* (one_vec .- y) .* node.gradient)'
end

binarycrossentropy(ŷ, y) = BroadcastedOperator(binarycrossentropy, ŷ::GraphNode, y::GraphNode)
forward!(node::BroadcastedOperator{typeof(binarycrossentropy)}, ŷ, y) = begin
    node.output .= Float32[sum((-(y .* log.(ŷ .+ ϵ))) .- (1.0f0 .- y) .* (log.(1.0f0 .- ŷ .+ ϵ)))./Float32(length(y))]
end
backward!(node::BroadcastedOperator{typeof(binarycrossentropy)}, ŷ_node::GraphNode, y_node::GraphNode) = begin
    ŷ = ŷ_node.output
    y = y_node.output

    grad_scaled = node.gradient / length(y)

    var_diff = 1.0f0 .- ŷ .+ ϵ
    y_node.gradient .= grad_scaled .* (log.(var_diff) .- log.(ŷ .+ ϵ))

    inv_var_diff = 1.0f0 ./ var_diff
    inv_ŷ = 1.0f0 ./ (ŷ .+ ϵ)
    ŷ_node.gradient .= grad_scaled .* (inv_var_diff .* (1.0f0 .- y) .- inv_ŷ .* y)
end

@inline function conv_backward_pass!(data::Array{Float32,3}, result::Array{Float32,3}, kernel::Array{Float32,3})
    data_rows, data_cols, _ = size(data)
    kernel_rows, _, kernel_depth = size(kernel)

    @inbounds @views @threads for row_idx in 1:(data_rows-kernel_rows+1)
        acc = zeros(Float32, data_cols, kernel_depth)
        for kernel_row in 1:kernel_rows
            mul!(acc, data[row_idx+kernel_row-1, :, :], kernel[kernel_row, :, :], 1.0f0, 1.0f0)
        end
        result[end-row_idx+1, :, :] .= acc
    end

    return result
end

@inline function im2col(A::SubArray{Float32,2,Array{Float32,3},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},Int64},true}, n::Int64, m::Int64)
    M, N = size(A)
    B = Matrix{Float32}(undef, m * n, (M - m + 1) * (N - n + 1))
    @inbounds indx = reshape(1:M*N, M, N)[1:M-m+1, 1:N-n+1]
    for (i, value) in enumerate(indx)
        for j = 0:n-1
            @inbounds @views B[(i-1)*m*n+j*m+1:(i-1)m*n+(j+1)m] = A[value+j*M:value+m-1+j*M]
        end
    end
    return B'
end

@inline function conv!(one::Array{Float32,3}, two::Array{Float32,3}, response::Array{Float32,3}, flipped=false)
    if (!flipped)
        two = reverse(two, dims=1)
    end
    kernel = reshape(two, :, size(two, 3))
    @threads for i in 1:size(one, 3)
        @inbounds @views response[:, :, i] = im2col(one[:, :, i], size(one, 2), size(two, 1)) * kernel
    end
end

convolution(y, w) = BroadcastedOperator(convolution, y::GraphNode, w::GraphNode; storage=zeros(Float32, 2 * size(w.output, 1) - 1 + size(y.output, 1) - size(w.output, 1), size(w.output, 3), size(y.output, 3)))
forward!(node::BroadcastedOperator{typeof(convolution)}, val, w) = begin
    conv!(val, w, node.output)
end
backward!(node::BroadcastedOperator{typeof(convolution)}, val::GraphNode, w::GraphNode) = begin
    w_permuted = permutedims(w.output, (1, 3, 2))

    @inbounds node.storage[size(w_permuted, 1):end-size(w_permuted, 1)+1, :, :] .= node.gradient

    conv!(node.storage, w_permuted, val.gradient, true)
    grad_perm = permutedims(node.gradient, (1, 3, 2))

    conv_backward_pass!(val.output, w.gradient, grad_perm)
end

@inline function gather_backward_pass!(x::Matrix{Int32}, w::Matrix{Float32}, grad::Array{Float32,3})
    @threads for i in 1:size(x, 2)
        @inbounds for j in 1:size(x, 1)
            col = x[j, i]
            @simd for k in 1:size(w, 1)
                w[k, col] += grad[j, k, i]
            end
        end
    end
end

@inline function gather!(x::Matrix{Int32}, w::Matrix{Float32}, result::Array{Float32,3})
    @threads for i in 1:size(x, 2)
        @inbounds for j in 1:size(x, 1)
            result[j, :, i] .= @view w[:, x[j, i]]
        end
    end
end

embedding(x, w) = BroadcastedOperator(embedding, x::GraphNode, w::GraphNode)
forward!(node::BroadcastedOperator{typeof(embedding)}, x, w) = begin
    gather!(x, w, node.output)
end
backward!(node::BroadcastedOperator{typeof(embedding)}, x::GraphNode, w::GraphNode) = begin
    gather_backward_pass!(x.output, w.gradient, node.gradient)
end

@inline function maxpool_with_mask!(mat::Array{Float32,3}, pooled::Array{Float32,3}, mask_storage::Array{Float32,3}, pool_size::Int64)
    x_len, y_len, z_len = size(mat)
    pooled_len = Int64(x_len / pool_size)
    fill!(mask_storage, 0.0f0)
    @threads for channel in 1:z_len
        for col in 1:y_len
            for block in 1:pooled_len
                start_idx = (block - 1) * pool_size + 1
                end_idx = block * pool_size
                val, idx = findmax(@inbounds @view mat[start_idx:end_idx, col, channel])
                @inbounds pooled[block, col, channel] = val
                @inbounds mask_storage[start_idx+idx[1]-1, col, channel] = 1.0f0
            end
        end
    end
    return pooled
end

maxpool(x, pool_size) = BroadcastedOperator(maxpool, x::GraphNode, pool_size::GraphNode)
forward!(node::BroadcastedOperator{typeof(maxpool)}, x, pool_size) = begin
    maxpool_with_mask!(x, node.output, node.inputs[1].gradient, pool_size[1])
end
backward!(node::BroadcastedOperator{typeof(maxpool)}, x::GraphNode, pool_size::GraphNode) = begin
    step = pool_size.output[1]
    @threads for i in 1:size(node.gradient, 1)
        start_idx = (i - 1) * step + 1
        end_idx = i * step
        @inbounds @views x.gradient[start_idx:end_idx, :, :] .*= node.gradient[i:i, :, :]
    end
end

flatten(x) = BroadcastedOperator(flatten, x::GraphNode)
forward!(node::BroadcastedOperator{typeof(flatten)}, x) = begin
    node.output .= reshape(x, :, size(x)[end])
end
backward!(node::BroadcastedOperator{typeof(flatten)}, x::GraphNode) = begin
    x.gradient .= reshape(node.gradient, size(x.output))
end