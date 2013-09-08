module LeNet5

using Base.Test

type C1
    weights
    biases
end

function run(layer::C1, input)
    @assert size(input) == (32,32)
    output = zeros(6,28,28)
    for feature_map in 1:6
        for i in 1:28
            for j in 1:28
                output[feature_map, i, j] = squash(sum(input[i:i+5-1, j:j+5-1] .* layer.weights[feature_map]) + layer.biases[feature_map])
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
    c1 = C1(rand(6, 5,5), rand(6))

    output = run(c1, input)
    @test size(output) == (6,28,28)
    @test output[1, 1, 1] == squash(sum(input[1:5, 1:5] .* c1.weights[1]) + c1.biases[1])
    @test output[3, 5, 2] == squash(sum(input[5:9, 2:6] .* c1.weights[3]) + c1.biases[3])
end

end

