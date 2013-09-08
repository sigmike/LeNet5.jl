module LeNet5

using Base.Test

type C1
    weights
    biases
end
C1() = C1(random_weight(32*32, 6, 5,5), random_weight(32*32, 6))

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
    c1 = C1()
    @test valid_weight(c1.weights, 32*32)
    @test valid_weight(c1.biases, 32*32)

    output = run(c1, input)
    @test size(output) == (6,28,28)
    @test output[1, 1, 1] == squash(sum(input[1:5, 1:5] .* c1.weights[1]) + c1.biases[1])
    @test output[3, 5, 2] == squash(sum(input[5:9, 2:6] .* c1.weights[3]) + c1.biases[3])
end

type S2
    coefficient
    bias
end

random_weight(inputs) = (rand() * 2 - 1) * 2.4 / inputs
random_weight(inputs, size...) = (rand(size...) .* 2 - 1) .* 2.4 ./ inputs

S2() = S2(random_weight(28*28), random_weight(28*28))

function run(layer::S2, input)
    output = zeros(6, 14, 14)
    for f in 1:6, i in 1:14, j in 1:14
        i2 = i*2-1
        j2 = j*2-1
        output[f, i, j] = sigmoid((input[f, i2, j2] + input[f, i2+1, j2] + input[f, i2, j2+1] + input[f, i2+1, j2+1]) * layer.coefficient + layer.bias)
    end
    output
end

sigmoid(x) = 1 / (1 + e^-x)

valid_weight(w::Number, inputs) = (max = 2.4/inputs; w <= max && w >= -max)
valid_weight(w::Array, inputs) = (max = 2.4/inputs; all((x) -> valid_weight(x, inputs), w) && abs(std(w) - max / 2) <= 0.01 )

function test_s2()
    srand(123)
    input = rand(6, 28,28)
    s2 = S2()
    @test valid_weight(s2.coefficient, 28*28)
    @test valid_weight(s2.bias, 28*28)

    output = run(s2, input)
    @test size(output) == (6, 14, 14)
    @test output[1, 1, 1] == sigmoid((input[1,1,1] + input[1,1,2] + input[1,2,1] + input[1,2,2]) * s2.coefficient + s2.bias)
    @test output[3, 5, 7] == sigmoid((input[3,9,13] + input[3,9,14] + input[3,10,13] + input[3,10,14]) * s2.coefficient + s2.bias)
end

end

