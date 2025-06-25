using Base.Threads

struct Adam
    eta::Float32
    beta::Tuple{Float32,Float32}
    betaP::Vector{Float32}
    ϵ::Float32
    mt::Vector{Union{Array{Float32,3},Matrix{Float32},Vector{Float32}}}
    vt::Vector{Union{Array{Float32,3},Matrix{Float32},Vector{Float32}}}

    function Adam(opt_values::Vector{Variable}, η::Float32=0.001f0, β::Tuple{Float32,Float32}=(0.9f0, 0.999f0), ϵ::Float32=Float32(1e-8))
        mt = Vector{Union{Array{Float32,3},Matrix{Float32},Vector{Float32}}}()
        vt = Vector{Union{Array{Float32,3},Matrix{Float32},Vector{Float32}}}()
        for el in opt_values
            push!(mt, zeros(Float32, size(el.output)))
            push!(vt, zeros(Float32, size(el.output)))
        end
        new(η, (β[1], β[2]), [β[1], β[2]], ϵ, mt, vt)
    end
end

@inline function apply_large_matrix!(el, mt, vt,
    β1, β2, one_minus_beta1, one_minus_beta2,
    inv_one_minus_betaP1, inv_one_minus_betaP2,
    η, ϵ)

    output = el.output
    grad = el.gradient

    @threads for j in axes(output, 2)
        @inbounds for i in axes(output, 1)
            g = grad[i, j]
            m = mt[i, j] = β1 * mt[i, j] + one_minus_beta1 * g
            v = vt[i, j] = β2 * vt[i, j] + one_minus_beta2 * g * g
            denom = sqrt(v * inv_one_minus_betaP2) + ϵ
            output[i, j] -= η * m * inv_one_minus_betaP1 / denom
        end
    end
end

function apply!(o::Adam, opt_values::Vector{Variable})
    η, β, βP = o.eta, o.beta, o.betaP
    β1, β2 = β
    βP1, βP2 = βP
    one = 1.0f0

    one_minus_beta1 = one - β1
    one_minus_beta2 = one - β2
    inv_one_minus_betaP1 = one / (one - βP1)
    inv_one_minus_betaP2 = one / (one - βP2)

    @inbounds @simd for i in 1:length(opt_values)
        el = opt_values[i]
        mt = o.mt[i]
        vt = o.vt[i]
        if size(el.output) == (50, 12849)
            apply_large_matrix!(
                el, mt, vt,
                β1, β2, one_minus_beta1, one_minus_beta2, inv_one_minus_betaP1, inv_one_minus_betaP2,
                η, o.ϵ
            )
        else
            @. mt = β1 * mt + one_minus_beta1 * el.gradient
            @. vt = β2 * vt + one_minus_beta2 * el.gradient * el.gradient
            @. el.output -= mt * inv_one_minus_betaP1 / (√(vt * inv_one_minus_betaP2) + o.ϵ) * η
        end
    end

    βP .= βP .* β
end
