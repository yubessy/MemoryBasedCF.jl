using SparseArrays

struct Config
    centering::Bool

    function Config(;centering = true)
        new(centering)
    end
end

const defaultconfig = Config()

struct Memory
    nu::Int # number of users
    ni::Int # number of items
    bu::Vector{Float64} # user biases
    bi::Vector{Float64} # item biases
    Dui::SparseMatrixCSC{Float64,Int} # user-item residuals
    Diu::SparseMatrixCSC{Float64,Int} # item-user residuals
    Suu::SparseMatrixCSC{Float64,Int} # user-user similarities
    Sii::SparseMatrixCSC{Float64,Int} # item-item similarities
end

function memorize(
    Rui::SparseMatrixCSC{Float64,Int},
    config::Config = defaultconfig,
)::Memory
    """construct memory for recommendation"""
    nu, ni = size(Rui)
    Riu = sparse(Rui')
    if config.centering
        bu = biases(Riu)
        bi = biases(Rui)
        Dui = centering(Rui, bu)
        Diu = centering(Riu, bi)
    else
        bu = zeros(nu)
        bi = zeros(ni)
        Dui = Rui
        Diu = Riu
    end
    Sii = cossim(Dui)
    Suu = cossim(Diu)
    Memory(nu, ni, bu, bi, Dui, Diu, Suu, Sii)
end

function itembased_scores(
    m::Memory,
    users::Vector{Int},
    target_items::Union{Nothing,Vector{Int}} = nothing,
)::Matrix{Float64}
    """user-item scores based on item similarities"""
    items = target_items == nothing ? collect(1:m.ni) : target_items
    scores(m.bu, m.Dui, m.Sii, users, items)
end

function userbased_scores(
    m::Memory,
    users::Vector{Int},
    target_items::Union{Nothing,Vector{Int}} = nothing,
)::Matrix{Float64}
    """user-item scores based on user similarities"""
    items = target_items == nothing ? collect(1:m.ni) : target_items
    scores(m.bi, m.Diu, m.Suu, items, users)'
end

function itembased_rankings(
    m::Memory,
    k::Int,
    users::Vector{Int},
    target_items::Union{Nothing,Vector{Int}} = nothing,
)::Tuple{Matrix{Int}, Matrix{Float64}}
    """rank top-k items for users based on item similarities and return them with scores"""
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
    target_items::Union{Nothing,Vector{Int}} = nothing,
)::Tuple{Matrix{Int}, Matrix{Float64}}
    """rank top-k items for users based on user similarities and return them with scores"""
    items = target_items == nothing ? collect(1:m.ni) : target_items
    @assert k <= length(items)
    scores = userbased_scores(m, users, items)
    perms = topkperm(scores, k)
    items[perms], selectbyrow(scores, perms)
end

function scores(
    b::Vector{Float64},
    D::SparseMatrixCSC{Float64,Int},
    S::SparseMatrixCSC{Float64,Int},
    xs::Vector{Int},
    ys::Vector{Int},
)::Matrix{Float64}
    """calc scores of users x items"""
    b = b[xs]
    D = D[xs, :]
    S = S[:, ys]

    is, js, vs = findnz(D)
    I = sparse(is, js, ones(length(vs)))
    N = abs.(S)
    W = 1. ./ (I * N)
    W[isinf.(W)] .= 0.
    W = sparse(W)

    b .+ W .* (D * S)
end

function biases(R::SparseMatrixCSC{Float64,Int})::Vector{Float64}
    """biases for each columns"""
    sum(R, dims = 1)[:] ./ mapslices(nnz, R, dims = 1)[:]
end

function centering(
    R::SparseMatrixCSC{Float64,Int},
    b::Vector{Float64},
)::SparseMatrixCSC{Float64,Int}
    """centering each values"""
    xs, ys, vs = findnz(R)
    sparse(xs, ys, vs - b[xs])
end

function cossim(D::SparseMatrixCSC{Float64,Int})::SparseMatrixCSC{Float64,Int}
    """calc cosine similarity between each columns"""
    # normalizing weights for each columns
    w = 1. ./ sqrt.(sum(D .^ 2, dims = 1)[:])
    w[isinf.(w)] .= 0.
    w = sparsevec(w)

    # normalizing D
    wD = w' .* D

    # calc similarities
    wD' * wD
end

function topkperm(V::Matrix{Float64}, k::Int)::Matrix{Int}
    """select top-k indices by each row of matrix"""
    mapslices(vs -> partialsortperm(vs, 1:k, rev = true), V, dims = 2)
end

function selectbyrow(M::Matrix{Float64}, I::Matrix{Int})::Matrix{Float64}
    """select matrix contents using indices by each row"""
    vcat((M[i, :][I[i, :]]' for i in 1:size(I, 1))...)
end
