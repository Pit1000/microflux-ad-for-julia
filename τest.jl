push!(LOAD_PATH, pwd())

using JLD2
using Test  
using μFlux

@testset "Automatic Differentiation Tests" begin

    @testset "Scalar: Power (x^c)" begin
        x = Variable(3.0f0, name="x")
        c = Constant(4.0f0)
        y = x ^ c

        graph = topological_sort(y)
        y_val = forward!(graph)
        backward!(graph)

        expected_y = 3.0f0^4.0f0
        expected_grad_x = 4.0f0 * 3.0f0^3.0f0

        @test isapprox(y_val, expected_y, atol=1e-6)
        @test isapprox(x.gradient, expected_grad_x, atol=1e-6)
    end

    @testset "Scalar: Sin(x)" begin
        x = Variable(pi/2.0f0, name="x")
        y = sin(x)

        graph = topological_sort(y)
        y_val = forward!(graph)
        backward!(graph)

        expected_y = sin(pi/2.0f0)
        expected_grad_x = cos(pi/2.0f0)

        @test isapprox(y_val, expected_y, atol=1e-7)
        @test isapprox(x.gradient, expected_grad_x, atol=1e-7)
    end

    @testset "Broadcast: Matrix Multiplication (*)" begin
        A = Variable(Float32[1 2; 3 4], name="A")
        x = Variable(Float32[5; 6], name="x")
        y = A * x

        graph = topological_sort(y)
        y_val = forward!(graph)
        backward!(graph)

        expected_y = Float32[17; 39]
        expected_grad_A = [1.0f0; 1.0f0] * transpose(x.output) # g * x'
        expected_grad_x = transpose(A.output) * [1.0f0; 1.0f0] # A' * g

        @test isapprox(y_val, expected_y)
        @test isapprox(A.gradient, expected_grad_A)
        @test isapprox(x.gradient, expected_grad_x)
    end

    @testset "Broadcast: Element-wise Multiplication (.*)" begin
        x = Variable(Float32[1 2; 3 4], name="x")
        y = Variable(Float32[5 6; 7 8], name="y")
        z = x .* y

        graph = topological_sort(z)
        y_val = forward!(graph)
        backward!(graph)

        expected_y = Float32[5 12; 21 32]
        expected_grad_x = y.output # g .* y, 
        expected_grad_y = x.output # g .* x, 

        @test isapprox(y_val, expected_y)
        @test isapprox(x.gradient, expected_grad_x)
        @test isapprox(y.gradient, expected_grad_y)
    end

    @testset "Layer: Conv1D" begin
        x_val = reshape([1., 2., 3., 4.], 4, 1, 1) # W=4, C_in=1, N=1
        K_val = reshape([0.1, 0.2, 0.3], 3, 1, 1) # KW=3, C_in=1, C_out=1
        b_val = [10.] # C_out=1
        
        x = Variable(Float32.(x_val), name="x")
        K = Variable(Float32.(K_val), name="K")
        b = Variable(Float32.(b_val), name="b")
        y = Conv1DNode(x, K, b)
        
        graph = topological_sort(y)
        y_val = forward!(graph)
        backward!(graph)

        # Ręczne obliczenia kroku forward
        y1 = (1*0.1 + 2*0.2 + 3*0.3) + 10  # = 1.4 + 10 = 11.4
        y2 = (2*0.1 + 3*0.2 + 4*0.3) + 10  # = 2.0 + 10 = 12.0
        expected_y = reshape([y1, y2], 2, 1, 1)
        
        # Ręczne obliczenia kroku backward (seed g=1)
        # g_b = sum(g) = 1+1=2
        # g_K[0] = g[0]*x[0] + g[1]*x[1] = 1*1+1*2 = 3
        # g_K[1] = g[0]*x[1] + g[1]*x[2] = 1*2+1*3 = 5
        # g_K[2] = g[0]*x[2] + g[1]*x[3] = 1*3+1*4 = 7
        # g_x[0] = g[0]*K[0] = 1*0.1 = 0.1
        # g_x[1] = g[0]*K[1] + g[1]*K[0] = 1*0.2+1*0.1 = 0.3
        # g_x[2] = g[0]*K[2] + g[1]*K[1] = 1*0.3+1*0.2 = 0.5
        # g_x[3] = g[1]*K[2] = 1*0.3 = 0.3
        expected_g_x = reshape([0.1, 0.3, 0.5, 0.3], 4, 1, 1)
        expected_g_K = reshape([3., 5., 7.], 3, 1, 1)
        expected_g_b = [2.]

        @test isapprox(y_val, Float32.(expected_y), atol=1e-6)
        @test isapprox(x.gradient, Float32.(expected_g_x), atol=1e-6)
        @test isapprox(K.gradient, Float32.(expected_g_K), atol=1e-6)
        @test isapprox(b.gradient, Float32.(expected_g_b), atol=1e-6)
    end

    @testset "Expr: sin(x^2)" begin
        x = Variable(5.0, name="x")
        two = Constant(2.0)
        squared = x^two
        sine = sin(squared)

        graph = topological_sort(sine)
        y_val = forward!(graph)
        backward!(graph)

        expected_y = sin(25.0)
        expected_grad = 2 * 5.0 * cos(25.0)

        @test isapprox(y_val, expected_y; atol=1e-10)
        @test isapprox(x.gradient, expected_grad; atol=1e-10)
    end

    @testset "Expr: sum[​ReLU(Wx+b)]" begin
        # Dane wejściowe 
        x_val = Float32[0.5; 1.0] 
        
        # Wagi 
        W_val = Float32[
            1.0  2.0; 
            -1.0 0.5; 
            0.0  -2.0
        ] 
        
        # Biasy 
        b_val = Float32[0.1; 0.2; -0.3]

        x = Constant(x_val) 
        W = Variable(W_val, name="W") 
        b = Variable(b_val, name="b") 

        # --- forward ---
        z_node = W * x
        h_node = z_node .+ b
        a_node = relu(h_node)
        loss_node = sum(a_node)

        # --- backward  ---
        graph = topological_sort(loss_node)
        final = forward!(graph)
        backward!(graph) # seed=1.0 dla sumy

        # --- Ręczne obliczenia do weryfikacji  ---
        # forward
        z_out = W_val * x_val         # [2.5; 0.0; -2.0]
        h_out = z_out .+ b_val        # [2.6; 0.2; -2.3]
        a_out = max.(h_out, 0.0f0)    # [2.6; 0.2; 0.0]
        expected_loss = sum(a_out)    # 2.8

        # backward 
        g_loss = 1.0f0
        g_a = ones(Float32, size(a_out)) * g_loss 

        # Gradient dla relu(h)
        g_h = g_a .* (h_out .> 0.0f0) # [1.0, 1.0, 0.0]

        # Gradient dla h = z + b 
        g_z = g_h
        g_b = g_h # gradient dla biasów

        # Gradient dla z = W * x
        g_W = g_z * transpose(x_val) 
        
        @test isapprox(final, expected_loss, atol=1e-6)
        @test isapprox(b.gradient, g_b, atol=1e-6)
        @test isapprox(W.gradient, g_W, atol=1e-6)
        
    end

end