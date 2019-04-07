# MemoryBasedCF.jl

Memory-based collaborative filtering in Julia

## Usage

```julia
using SparseArrays
using MemoryBasedCF

# rating matrix
# zero means the user has not rated the item yet.
R = sparse([
    2.0  1.0  0.0
    1.0  0.0  2.0
    0.0  1.0  2.0
    0.0  0.0  2.0
])

# constuct memory
# calc item-item and user-user similarities inside
memory = memorize(R)

# get predicted scores of items for users [1,2] using item-item similarities
itembased_scores(memory, [1,2])
# 2×3 Array{Float64,2}:
# 2.0  1.0  1.5
# 1.0  1.5  2.0

# get top-2 item rankings for users [1,2] using item-item similarities
ranking, scores = itembased_rankings(memory, 2, [1,2])

rankings
# 2×2 Array{Int64,2}:
# 1  3
# 3  2

scores
# 2×2 Array{Float64,2}:
# 2.0  1.5
# 2.0  1.5
```
