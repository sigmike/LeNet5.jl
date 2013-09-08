module LeNet5

using Base.Test

function c1(input, weights, bias)
    @assert size(input) == (32,32)
    output = zeros(6,28,28)
    for feature_map in 1:6
        for i in 1:28
            for j in 1:28
                output[feature_map, i, j] = squash(sum(input[i:i+5-1, j:j+5-1] .* weights[feature_map]) + bias[feature_map])
            end
        end
    end
    output
end

# lecun-98 p.41
function squash(a)
    A = 1.7159
    S = 2 / 3
    A * tanh(S * a)
end
@test_approx_eq_eps 1.0 squash(1) 1e-5
@test_approx_eq_eps -1.0 squash(-1) 1e-5

function test_c1()
    srand(123)
    input = rand(32,32)
    weights = rand(6, 5,5)
    bias = rand(6)

    output = c1(input, weights, bias)
    @test size(output) == (6,28,28)
    @test output[1, 1, 1] == squash(sum(input[1:5, 1:5] .* weights[1]) + bias[1]) 
    @test output[3, 5, 2] == squash(sum(input[5:9, 2:6] .* weights[3]) + bias[3]) 
end

end

