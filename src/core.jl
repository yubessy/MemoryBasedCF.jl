using SparseArrays

struct Memory
    nu::Int # N of users
    ni::Int # N of items
    bu::Vector{Float64} # user biases
    bi::Vector{Float64} # item biases
    Rui::SparseMatrixCSC{Float64,Int} # user-item ratings
    Riu::SparseMatrixCSC{Float64,Int} # item-user ratings
    Suu::SparseMatrixCSC{Float64,Int} # user-user similarities
    Sii::SparseMatrixCSC{Float64,Int} # item-item similarities
end

function memorize(Rui::SparseMatrixCSC{Float64,Int})::Memory
    """construct memory for recommendation"""
    nu, ni = size(Rui)
    Riu = sparse(Rui')
    bu = biases(Riu)
    bi = biases(Rui)
    Rui = centering(Rui, bu)
    Riu = centering(Riu, bi)
    Sii = cossim(Rui)
    Suu = cossim(Riu)
    Memory(nu, ni, bu, bi, Rui, Riu, Suu, Sii)
end

function itembased_scores(
    m::Memory,
    users::Vector{Int},
    target_items::Union{Nothing,Vector{Int}} = nothing
)::Matrix{Float64}
    """user-item scores based on item similarities"""
    items = target_items == nothing ? collect(1:m.ni) : target_items
    scores(m.bu, m.Rui, m.Sii, users, items)
end

function userbased_scores(
    m::Memory,
    users::Vector{Int},
    target_items::Union{Nothing,Vector{Int}} = nothing
)::Matrix{Float64}
    """user-item scores based on user similarities"""
    items = target_items == nothing ? collect(1:m.ni) : target_items
    scores(m.bi, m.Riu, m.Suu, items, users)'
end

function itembased_rankings(
    m::Memory,
    k::Int,
    users::Vector{Int},
    target_items::Union{Nothing,Vector{Int}} = nothing
)::Tuple{Matrix{Int}, Matrix{Float64}}
    items = target_items == nothing ? collect(1:m.ni) : target_items
    @assert k <= length(items)
    scores = itembased_scores(m, users, items)
    perms = topkperm(scores, k)
    items[perms], selectbyrow(scores, perms)
end

function userbased_rankings(
    m::Memory,
    k::Int,
    users::Vector{Int},
    target_items::Union{Nothing,Vector{Int}} = nothing
)::Tuple{Matrix{Int}, Matrix{Float64}}
    items = target_items == nothing ? collect(1:m.ni) : target_items
    @assert k <= length(items)
    scores = userbased_scores(m, users, items)
    perms = topkperm(scores, k)
    items[perms], selectbyrow(scores, perms)
end

function biases(R::SparseMatrixCSC{Float64,Int})::Vector{Float64}
    """biases for each columns"""
    sum(R, dims = 1)[:] ./ mapslices(nnz, R, dims = 1)[:]
end

function centering(R::SparseMatrixCSC{Float64,Int}, bx::Vector{Float64})::SparseMatrixCSC{Float64,Int}
    """centering each values"""
    xs, ys, vs = findnz(R)
    sparse(xs, ys, vs - bx[xs])
end

function cossim(R::SparseMatrixCSC{Float64,Int})::SparseMatrixCSC{Float64,Int}
    """calc cosine similarity between each columns"""
    # normalizing weights for each columns
    w = 1. ./ sqrt.(sum(R .^ 2, dims = 1)[:])
    w[isinf.(w)] .= 0.
    w = sparsevec(w)

    # normalizing R
    wR = R .* w'

    # calc similarities
    wR' * wR
end

function scores(b, R, S, xs, ys)
    b = b[xs]
    R = R[xs, :]
    S = S[:, ys]

    is, js, vs = findnz(R)
    I = sparse(is, js, ones(length(vs)))
    N = abs.(S)
    W = 1. ./ (I * N)
    W[isinf.(W)] .= 0.
    W = sparse(W)

    b .+ W .* (R * S)
end

topkperm(V, k) = mapslices(vs -> partialsortperm(vs, 1:k, rev = true), V, dims = 2)

selectbyrow(M, I) = vcat((M[i, :][I[i, :]]' for i in 1:size(I, 1))...)
