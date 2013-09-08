module LeNet5

function c1(input)
    @assert size(input) == (32,32)
    zeros(28,28)
end

using Base.Test

function test_c1()
    @test size(c1(zeros(32,32))) == (28,28)
end

end

