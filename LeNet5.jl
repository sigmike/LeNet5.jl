module LeNet5

type InputLayer
    size
end

InputLayer(; size = Nothing) = InputLayer(size)

output_size(layer::InputLayer) = layer.size

type ConvolutionalLayer
    input_layer
    feature_maps
    neighborhood
end

ConvolutionalLayer(; input_layer = Nothing, feature_maps = Nothing, neighborhood = Nothing) = ConvolutionalLayer(input_layer, feature_maps, neighborhood)

function output_size(layer::ConvolutionalLayer)
    input_size = output_size(layer.input_layer)
    (
        input_size[1] - layer.neighborhood[1] + 1,
        input_size[2] - layer.neighborhood[2] + 1,
    )
end

function lenet5()
    input = InputLayer(size = (32,32))
    c1 = ConvolutionalLayer(input_layer = input, feature_maps = 6, neighborhood = (5,5))
    @assert output_size(c1) == (28,28)
end

end

