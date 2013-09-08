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

type SubSamplingLayer
    input_layer
    feature_maps
    neighborhood
end

function SubSamplingLayer(; input_layer = Nothing, feature_maps = Nothing, neighborhood = Nothing)
    SubSamplingLayer(input_layer, feature_maps, neighborhood)
end

function output_size(layer::SubSamplingLayer)
    input_size = output_size(layer.input_layer)
    (
        div(input_size[1], layer.neighborhood[1]),
        div(input_size[2], layer.neighborhood[2]),
    )
end

function lenet5()
    input = InputLayer(size = (32,32))
    c1 = ConvolutionalLayer(input_layer = input, feature_maps = 6, neighborhood = (5,5))
    @assert output_size(c1) == (28,28)

    s2 = SubSamplingLayer(input_layer = c1, feature_maps = 6, neighborhood = (2,2))
    @assert output_size(s2) == (14, 14)
end

end

