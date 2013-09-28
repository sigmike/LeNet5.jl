module LeNet5

using Base.Test

# lecun-98 p.41
random_weight(inputs) = (rand() * 2 - 1) * 2.4 / inputs
random_weight(inputs, size...) = (rand(size...) .* 2 - 1) .* 2.4 ./ inputs
valid_weight(w::Number, inputs) = (max = 2.4/inputs; w <= max && w >= -max)
valid_weight(w::Array, inputs) = (max = 2.4/inputs; all((x) -> valid_weight(x, inputs), w) && abs(std(w) - max / 2) <= 0.2 )

# lecun-98 p.8 and 41
A = 1.7159
S = 2 / 3
function squash(a)
    A * tanh(S * a)
end
@test_approx_eq_eps 1.0 squash(1) 1e-5
@test_approx_eq_eps -1.0 squash(-1) 1e-5
@test_approx_eq_eps 0.140843782 squash(0.1234) 1e-5

abstract Layer

type C1 <: Layer
    weights
    biases

    output
end
C1() = C1(random_weight(5*5, 6, 5,5), random_weight(5*5, 6), zeros(6,28,28))

function run(input, layer::C1)
    @assert size(input) == (32,32)
    for feature_map in 1:6
        for i in 1:28
            for j in 1:28
                layer.output[feature_map, i, j] = squash(sum(input[i:i+5-1, j:j+5-1] .* layer.weights[feature_map]) + layer.biases[feature_map])
            end
        end
    end
    layer.output
end

parameters(c1::C1) = [c1.weights..., c1.biases...]

function test_c1()
    srand(123)
    input = rand(32,32)
    c1 = C1()
    @test valid_weight(c1.weights, 5*5)
    @test valid_weight(c1.biases, 5*5)
    @test size(parameters(c1)) == (156,)

    output = run(input, c1)
    @test size(output) == (6,28,28)
    @test output[1, 1, 1] == squash(sum(input[1:5, 1:5] .* c1.weights[1]) + c1.biases[1])
    @test output[3, 5, 2] == squash(sum(input[5:9, 2:6] .* c1.weights[3]) + c1.biases[3])
end

type S2 <: Layer
    coefficients
    biases

    output
end
S2() = S2(random_weight(2*2,6), random_weight(2*2,6), zeros(6, 14, 14))

function run(input, layer::S2)
    for f in 1:6, i in 1:14, j in 1:14
        i2 = i*2-1
        j2 = j*2-1
        layer.output[f, i, j] = squash((input[f, i2, j2] + input[f, i2+1, j2] + input[f, i2, j2+1] + input[f, i2+1, j2+1]) * layer.coefficients[f] + layer.biases[f])
    end
    layer.output
end
parameters(s2::S2) = [s2.coefficients..., s2.biases...]

function test_s2()
    srand(123)
    input = rand(6, 28,28)
    s2 = S2()
    @test valid_weight(s2.coefficients, 2*2)
    @test valid_weight(s2.biases, 2*2)
    @test size(parameters(s2)) == (12,)

    output = run(input, s2)
    @test size(output) == (6, 14, 14)
    @test output[1, 1, 1] == squash((input[1,1,1] + input[1,1,2] + input[1,2,1] + input[1,2,2]) * s2.coefficients[1] + s2.biases[1])
    @test output[3, 5, 7] == squash((input[3,9,13] + input[3,9,14] + input[3,10,13] + input[3,10,14]) * s2.coefficients[3] + s2.biases[3])
end

type C3FeatureMap
    input_map_indexes
    weights
    bias
end

type C3 <: Layer
    feature_maps

    output
end
function C3()
    maps = Array(C3FeatureMap, (16))
    index_maps = { 
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [0, 4, 5],
        [0, 1, 5],
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [0, 3, 4, 5],
        [0, 1, 4, 5],
        [0, 1, 2, 5],
        [0, 1, 3, 4],
        [1, 2, 4, 5],
        [0, 2, 3, 5],
        [0, 1, 2, 3, 4, 5],
    }
    for i in 1:size(index_maps)[1]
        index_map = index_maps[i] + 1
        map = C3FeatureMap(index_map, random_weight(length(index_map)*5*5, length(index_map), 5, 5), random_weight(length(index_map)*5*5))
        maps[i] = map 
    end
    C3(maps, zeros(16, 10, 10))
end
function parameters(layer::C3)
    result = {}
    for map in layer.feature_maps
        append!(result, [map.weights..., map.bias])
    end
    result
end
function run(input, layer::C3)
    @assert size(input) == (6, 14, 14)
    fill!(layer.output, 0)
    for feature_map_index in 1:16
        feature_map = layer.feature_maps[feature_map_index]
        for i in 1:10
            for j in 1:10
                for input_index in 1:length(feature_map.input_map_indexes)
                    input_map_index = feature_map.input_map_indexes[input_index]
                    layer.output[feature_map_index, i, j] += sum(input[input_map_index, i:i+5-1, j:j+5-1] .* feature_map.weights[input_index, 1:5, 1:5])
                end
                layer.output[feature_map_index, i, j] = squash(layer.output[feature_map_index, i, j] + feature_map.bias)
            end
        end
    end
    layer.output
end

function test_c3()
    c3 = C3()
    @test size(parameters(c3)) == (1516,)
    @test valid_weight(c3.feature_maps[1].weights, 3*5*5)
    @test valid_weight(c3.feature_maps[8].weights, 4*5*5)

    input = rand(6, 14, 14)
    output = run(input, c3)

    @test size(output) == (16, 10, 10)
    @test output[1, 1, 1] == squash(
        sum(input[1, 1:5, 1:5] .* c3.feature_maps[1].weights[1, 1:5, 1:5]) +
        sum(input[2, 1:5, 1:5] .* c3.feature_maps[1].weights[2, 1:5, 1:5]) +
        sum(input[3, 1:5, 1:5] .* c3.feature_maps[1].weights[3, 1:5, 1:5]) +
        c3.feature_maps[1].bias)
    @test output[8, 3, 5] == squash(
        sum(input[2, 3:7, 5:9] .* c3.feature_maps[8].weights[1, 1:5, 1:5]) +
        sum(input[3, 3:7, 5:9] .* c3.feature_maps[8].weights[2, 1:5, 1:5]) +
        sum(input[4, 3:7, 5:9] .* c3.feature_maps[8].weights[3, 1:5, 1:5]) +
        sum(input[5, 3:7, 5:9] .* c3.feature_maps[8].weights[4, 1:5, 1:5]) +
        c3.feature_maps[8].bias)
end

type S4 <: Layer
    coefficients
    biases

    output
end
S4() = S4(random_weight(2*2,16), random_weight(2*2,16), zeros(16, 5, 5))

function run(input, layer::S4)
    for f in 1:16, i in 1:5, j in 1:5
        i2 = i*2-1
        j2 = j*2-1
        layer.output[f, i, j] = squash((input[f, i2, j2] + input[f, i2+1, j2] + input[f, i2, j2+1] + input[f, i2+1, j2+1]) * layer.coefficients[f] + layer.biases[f])
    end
    layer.output
end
parameters(s4::S4) = [s4.coefficients..., s4.biases...]

function test_s4()
    srand(123)
    input = rand(16,10,10)
    layer = S4()
    @test valid_weight(layer.coefficients, 2*2)
    @test valid_weight(layer.biases, 2*2)
    @test size(parameters(layer)) == (32,)

    output = run(input, layer)
    @test size(output) == (16, 5, 5)
    @test output[1, 1, 1] == squash((input[1,1,1] + input[1,1,2] + input[1,2,1] + input[1,2,2]) * layer.coefficients[1] + layer.biases[1])
    @test output[3, 5, 2] == squash((input[3,9,3] + input[3,9,4] + input[3,10,3] + input[3,10,4]) * layer.coefficients[3] + layer.biases[3])
end

type C5FeatureMap
    weights
    bias
end

type C5 <: Layer
    feature_maps

    output
end
function C5()
    maps = Array(C5FeatureMap, (120,))
    for i in 1:120
        map = C5FeatureMap(random_weight(16*5*5, 16, 5, 5), random_weight(16*5*5))
        maps[i] = map 
    end
    C5(maps, zeros(120, 1, 1))
end
function parameters(layer::C5)
    result = {}
    for map in layer.feature_maps
        append!(result, [map.weights..., map.bias])
    end
    result
end

function run(input, layer::C5)
    @assert size(input) == (16, 5, 5)
    fill!(layer.output, 0)
    for feature_map_index in 1:120
        feature_map = layer.feature_maps[feature_map_index]
        for i in 1:1
            for j in 1:1
                for input_index in 1:16
                    layer.output[feature_map_index, i, j] += sum(input[input_index, i:i+5-1, j:j+5-1] .* feature_map.weights[input_index, 1:5, 1:5])
                end
                layer.output[feature_map_index, i, j] = squash(layer.output[feature_map_index, i, j] + feature_map.bias)
            end
        end
    end
    layer.output
end

function test_c5()
    srand(1223)
    input = rand(16, 5, 5)
    layer = C5()
    @test length(parameters(layer)) == 48120
    @test valid_weight(layer.feature_maps[1].weights, 16*5*5)
    @test valid_weight(layer.feature_maps[75].weights, 16*5*5)

    output = run(input, layer)
    @test size(output) == (120,1,1)
    @test_approx_eq_eps output[1, 1, 1] squash(sum(input .* layer.feature_maps[1].weights) + layer.feature_maps[1].bias) 1e-10
    @test_approx_eq_eps output[75, 1, 1] squash(sum(input .* layer.feature_maps[75].weights) + layer.feature_maps[75].bias) 1e-10
end

type F6 <: Layer
    weights
    biases

    output
end
F6() = F6(random_weight(120, 84, 120), random_weight(120, 84), zeros(84))

function parameters(layer::F6)
    [layer.weights..., layer.biases...]
end

function run(input, layer::F6)
    for i in 1:84
        layer.output[i] = squash(sum(input .* layer.weights[i]) + layer.biases[i])
    end
    layer.output
end

function test_f6()
    srand(123)
    input = rand(120,1,1)
    layer = F6()
    @test valid_weight(layer.weights, 120)
    @test valid_weight(layer.biases, 120)
    @test length(parameters(layer)) == 10164

    output = run(input, layer)
    @test size(output) == (84,)
    @test output[1] == squash(sum(input .* layer.weights[1]) + layer.biases[1])
    @test output[79] == squash(sum(input .* layer.weights[79]) + layer.biases[79])
end

type Output <: Layer
    weights

    output
end

using Images

function Output()
    weights = zeros(96, 12, 7)

    for i in 1:96
        filename = "output_rbfs/$(lpad(i,3,'0')).ppm"
        image = imread(filename)
        array = convert(Array, image)[:,:,1]'
        for y in 1:size(array)[1]
            for x in 1:size(array)[2]
                weights[i,y,x] = array[y, x]  == 0 ? 1 : -1
            end
        end
    end

    Output(weights, zeros(96))
end

function run(input, layer::Output)
    fill!(layer.output, 0)
    for i in 1:96
        for j in 1:84
            layer.output[i] += ((input[j] - layer.weights[i,j]) .^ 2)
        end
    end
    layer.output
end

function test_output()
    srand(123)
    input = rand(84)
    layer = Output()
    @test reshape(layer.weights[ 2, 1, :], 7) == [-1, -1, -1,  1,  1, -1, -1]
    @test reshape(layer.weights[23, 9, :], 7) == [ 1,  1,  1, -1, -1,  1,  1]

    output = run(input, layer)
    @test size(output) == (96,)
    @test_approx_eq_eps output[ 1] sum((input - reshape(layer.weights[ 1,:,:], 84)).^2) 1e-10
    @test_approx_eq_eps output[21] sum((input - reshape(layer.weights[21,:,:], 84)).^2) 1e-10
end

type NeuralNetwork <: Layer
    c1
    s2
    c3
    s4
    c5
    f6
    output
end
NeuralNetwork() = NeuralNetwork(C1(), S2(), C3(), S4(), C5(), F6(), Output())

function run(input, network::NeuralNetwork)
    run(input, network.c1)
    run(network.c1.output, network.s2)
    run(network.s2.output, network.c3)
    run(network.c3.output, network.s4)
    run(network.s4.output, network.c5)
    run(network.c5.output, network.f6)
    run(network.f6.output, network.output)
    network.output.output
end

import Base.|>
(|>)(input, layer::Layer) = run(input, layer)

function test_lenet5()
    srand(123)
    input = rand(32, 32)
    lenet5 = NeuralNetwork()

    output = run(input, lenet5)

    expected_output = run(input, lenet5.c1)
    expected_output = run(expected_output, lenet5.s2)
    expected_output = run(expected_output, lenet5.c3)
    expected_output = run(expected_output, lenet5.s4)
    expected_output = run(expected_output, lenet5.c5)
    expected_output = run(expected_output, lenet5.f6)
    expected_output = run(expected_output, lenet5.output)
    @test output == expected_output
end

exp_minus_J = exp(-5)

function loss(outputs, desired_classes)
    training_samples = length(desired_classes)
    @assert size(outputs)[1] == training_samples
    errors = zeros(training_samples)
    for i in 1:training_samples
        errors[i] += outputs[i, desired_classes[i]]
        println("warning: still simplified loss function")
        #exp_outputs = map((x) -> exp(-x), outputs[i,:])
        #errors[i] += log(exp_minus_J + sum(exp_outputs))
    end
    mean(errors)
end

function test_loss()
    training_samples = 2
    network_outputs = ones(training_samples, 96)
    desired_output_classes = [6, 87]
    network_outputs[1, 6] = 0.1
    network_outputs[1, 7] = 0.5
    network_outputs[2, 87] = 0.5
    network_outputs[2, 12] = 0.2

    expected_error = 0.1 + log(exp_minus_J + exp(-0.1) + exp(-0.5) + 94 * exp(-1))
    expected_error += 0.5 + log(exp_minus_J + exp(-0.5) + exp(-0.2) + 94 * exp(-1))
    expected_error /= 2

    @test_approx_eq_eps loss(network_outputs, desired_output_classes) expected_error 1e-10
end

function test_loss_generic()
    srand(123)
    training_samples = 6
    network_outputs = rand(training_samples, 96)
    desired_output_classes = rand(1:96, training_samples)

    expected_error = 0
    for p in 1:training_samples
        expected_error += network_outputs[p, desired_output_classes[p]]
        expected_error += log(exp_minus_J + sum([exp(-output) for output in network_outputs[p,:]]))
    end
    expected_error /= training_samples
    @test loss(network_outputs, desired_output_classes) == expected_error
end

function backpropagate(error, layer, output, learning_rate)
end

function weighted_sum(layer, input, neuron_index)
    base_weights = layer.weights[neuron_index,:]

    input = reshape(input, length(input))
    weights = reshape(base_weights, length(base_weights))

    sum(input .* weights)
end

function derivative_of_weighted_sum_with_respect_to_weight(layer, input, neuron_index, connection_index)
    corresponding_input = input[connection_index, 1, 1]
end

function derivative_of_squash(x)
    A*(1 - tanh(S*x)^2) * S
end

function derivative_of_error_with_respect_to_F6_weight(input, network, desired_class, neuron_index, connection_index)
    run(input, network)
    c5_output = network.c5.output
    f6_output = network.f6.output
    network_output = run(f6_output, network.output)

    desired_class_weights = reshape(network.output.weights[desired_class,:,:], 84)

    # Error is calculated on a single training set. In the PDF the error is the average error of multiple training sets.
    #
    # Error derivative calculation:
    # error = sum((f6_output - desired_class_weights).^2) ## Simplified !
    @assert loss(reshape(network_output, 1, length(network_output)), [desired_class]) == sum((f6_output - desired_class_weights).^2)
    # derivative(error, w) = derivative(error, f6_output) * derivative(f6_output, neuron_weighted_sum) * derivative(neuron_weighted_sum, w)
    #
    # derivative(error, f6_output) = sum(derivative((f6_output - desired_class_weights).^2, f6_output))
    # derivative(error, f6_output) = sum(2(f6_output - desired_class_weights))
    #
    # f6_output = squash(neuron_weighted_sum + bias)
    # derivative(f6_output, neuron_weighted_sum) = derivative(f6_output, full_sum) * derivative(full_sum, neuron_weighted_sum)
    # derivative(full_sum, neuron_weighted_sum) = 1
    # derivative(squash, x) = derivative(squash(x), Sx) * derivative(Sx, x)
    # derivative(squash, x) = A*(1 - tanh(Sx)^2) * S # verified: http://math.stackexchange.com/questions/192433/derivative-of-neural-network-function
    # derivative(f6_output, full_sum) = A*(1 - tanh(S * full_sum)^2) * S
    #
    # derivative(neuron_weighted_sum, w) = corresponding_input
    # 
    # derivative(error, w) = sum(2(f6_output - desired_class_weights)) * (A*(1 - tanh(S*(neuron_weighted_sum + bias))^2) * S) * corresponding_input

    neuron_weighted_sum = weighted_sum(network.f6, c5_output, neuron_index)
    println(neuron_weighted_sum)
    bias = network.f6.biases[neuron_index]
    println(bias)
    corresponding_input = c5_output[connection_index, 1, 1]

    println((A*(1 - tanh(S*(neuron_weighted_sum))^2) * S))
    sum(2*(f6_output - desired_class_weights)) * (A*(1 - tanh(S*(neuron_weighted_sum + bias))^2) * S) * corresponding_input
end

using Winston

function show_derivative(range, value, setter, f, derivative_f)
    t = Nothing
    xs = range
    ys = zeros(length(xs))
    for i in 1:length(xs)
        setter(xs[i])
        ys[i] = f()
        if xs[i] == 0
            derivative = derivative_f()
            @show derivative
            t = [ys[i] + xx*derivative for xx in xs]
        end
    end
    @show ys

    p = FramedPlot()
    add(p, Curve(xs, ys))
    add(p, Curve(xs, t, "color", "red"))
    Winston.display(p)
end


function test_derivative_of_error_with_respect_to_F6_weight()
    srand(123)

    input = rand(32, 32)
    desired_class = 2

    network = NeuralNetwork()
    output = run(input, network)
    error = loss(reshape(output, 1, size(output)[1]), [desired_class])

    neuron_index = 1
    connection_index = 1

    network.f6.weights[neuron_index, connection_index] = 0

    derivative = derivative_of_error_with_respect_to_F6_weight(input, network, desired_class, neuron_index, connection_index)
    println(derivative)
  
    #show_derivative(-1:0.1:1, 0,
    #    (value)->begin
    #        network.f6.weights[neuron_index, connection_index] = value
    #        run(input, network)
    #    end,
    #    ()->weighted_sum(network.f6, network.c5.output, neuron_index),
    #    ()->derivative_of_weighted_sum_with_respect_to_weight(network.f6, network.c5.output, neuron_index, connection_index),
    #)

    #show_derivative(-1:0.1:1, 0,
    #    (value)->begin
    #        network.f6.weights[neuron_index, connection_index] = value
    #        run(input, network)
    #    end,
    #    ()->loss(reshape(network.output.output, 1, size(network.output.output)[1]), [desired_class]),
    #    ()->derivative_of_error_with_respect_to_F6_weight(input, network, desired_class, neuron_index, connection_index),
    #)

    x = 0
    show_derivative(-1:0.1:1, 0,
        (value)-> x = value,
        ()->squash(x),
        ()->derivative_of_squash(x),
    )

    change = 1.00
    network.f6.weights[neuron_index, connection_index] += change

    new_output = run(input, network)
    new_error = loss(reshape(new_output, 1, size(new_output)[1]), [desired_class])
    @test_approx_eq_eps new_error (error + derivative) 1e-10
end

function test_f6_backpropagation()
    srand(123)

    input = rand(32, 32)
    network = NeuralNetwork()

    c1_output = run(input, network.c1)
    s2_output = run(c1_output, network.s2)
    c3_output = run(s2_output, network.c3)
    s4_output = run(c3_output, network.s4)
    c5_output = run(s4_output, network.c5)
    f6_output = run(c5_output, network.f6)
    network_output = run(f6_output, network.output)

    desired_class = 25
    error = 5.0
    learning_rate = 0.1 # should be per neuron
    
    layer = network.f6
    neuron_index = 1
    weight_index = 1
    corresponding_input = c5_output[1,1,1]
    desired_class_weights = reshape(network.output.weights[desired_class,:,:], 84)
    weighted_sum = sum(c5_output .* layer.weights[neuron_index,:])
    bias = layer.biases[neuron_index]

    weight_change = -learning_rate * sum(2(f6_output - desired_class_weights)) * (1 - tanh(weighted_sum + bias)^2) * corresponding_input
   

    initial_weight = network.f6.weights[neuron_index,weight_index]
    expected_updated_weight = initial_weight + weight_change

    backpropagate(error, network.f6, f6_output, learning_rate)
    @test_approx_eq_eps network.f6.weights[neuron_index,weight_index] expected_updated_weight 1e-10
end

function test_all()
    test_derivative_of_error_with_respect_to_F6_weight()
    test_c1()
    test_s2()
    test_c3()
    test_s4()
    test_c5()
    test_f6()
    test_output()
    test_lenet5()
    test_loss()
    test_loss_generic()
    test_f6_backpropagation()
end

end
