using Test

using MemoryBasedCF
using SparseArrays

R = sparse([
    2.0  1.0  0.0
    1.0  0.0  2.0
    0.0  1.0  2.0
    0.0  0.0  2.0
])

@testset "default" begin
    m = memorize(R)

    @test m.nu == 4
    @test m.ni == 3
    @test m.bu == sparse([1.5, 1.5, 1.5, 2.0])
    @test m.bi == sparse([1.5, 1.0, 2.0])
    @test m.Rui == sparse([
         0.5  -0.5   0.0
        -0.5   0.0   0.5
         0.0  -0.5   0.5
         0.0   0.0   0.0
    ])
    @test m.Riu == sparse([
         0.5  -0.5   0.0   0.0
         0.0   0.0   0.0   0.0
         0.0   0.0   0.0   0.0
    ])
    @test isapprox(m.Sii, sparse([
         1.0  -0.5  -0.5
        -0.5   1.0  -0.5
        -0.5  -0.5   1.0
    ]), atol = 1e-2)
    @test isapprox(m.Suu, sparse([
         1.0  -1.0   0.0   0.0
        -1.0   1.0   0.0   0.0
         0.0   0.0   0.0   0.0
         0.0   0.0   0.0   0.0
    ]), atol = 1e-2)

    @testset "itembased" begin
        expected_scores = [
            2.25  0.75  1.50
            0.75  1.50  2.25
            1.50  0.75  2.25
            2.00  2.00  2.00
        ]
        @test isapprox(itembased_scores(m, [1, 2, 3, 4]), expected_scores, atol = 1e-2)
        @test isapprox(itembased_scores(m, [1, 2], [1, 3]), expected_scores[[1, 2], [1, 3]], atol = 1e-2)
    end

    @testset "userbased" begin
        expected_scores = [
            2.5  1.0  2.0
            0.5  1.0  2.0
            1.5  1.0  2.0
            1.5  1.0  2.0
        ]
        @test isapprox(userbased_scores(m, [1, 2, 3, 4]), expected_scores, atol = 1e-2)
        @test isapprox(userbased_scores(m, [1, 2], [1, 3]), expected_scores[[1, 2], [1, 3]], atol = 1e-2)
    end
end
