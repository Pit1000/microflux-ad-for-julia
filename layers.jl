"""
Struct for ADAM Optimizer

"""

mutable struct AdamOptimizer
    lr::Float32
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    t::Int
    m::IdDict{Variable, AbstractArray{Float32}}
    v::IdDict{Variable, AbstractArray{Float32}}
    function AdamOptimizer(lr::Real=0.001, beta1::Real=0.9, beta2::Real=0.999, epsilon::Real=1e-8)
        new(Float32(lr), Float32(beta1), Float32(beta2), Float32(epsilon), 0,
            IdDict{Variable, AbstractArray{Float32}}(), IdDict{Variable, AbstractArray{Float32}}())
    end
end

"""
Custom Layer/Model Representation

"""

struct DenseLayerConfig
    input_dims::Int; output_dims::Int; activation_graph_fn::Function
end
struct TrainingConfig
    epochs::Int; batch_size::Int; optimizer::AdamOptimizer
end

abstract type AbstractCustomLayer end
mutable struct DenseLayer <: AbstractCustomLayer
    W::Variable; 
    b::Variable; 
    activation_graph_fn::Function
end
mutable struct EmbeddingLayer <: AbstractCustomLayer
    W::Variable; 
    embedding_dim::Int; 
    vocab_size::Int
end
mutable struct Conv1DLayer <: AbstractCustomLayer
    K::Variable; 
    b::Variable; 
    activation_graph_fn::Function
end

mutable struct CustomModel
    layers::Vector{Union{AbstractCustomLayer, Function}}
    CustomModel(initial_layers::Vector) = new(convert(Vector{Union{AbstractCustomLayer, Function}}, initial_layers))
end

"""
DenseLayer, EmbeddingLayer, Conv1DLayer Constructor with weight glorot_uniform initialization 

"""

function DenseLayer(input_dims::Int, output_dims::Int, activation_graph_fn::Function; name_prefix::String="dense")
    W_val = glorot_uniform(output_dims, input_dims)
    b_val = zeros(Float32, output_dims, 1)
    W_var = Variable(W_val, name="$(name_prefix)_W")
    b_var = Variable(b_val, name="$(name_prefix)_b")
    return DenseLayer(W_var, b_var, activation_graph_fn)
end
function EmbeddingLayer(vocab_size::Int, embedding_dim::Int; name_prefix::String="embed")
    W_val = glorot_uniform(embedding_dim, vocab_size)
    W_var = Variable(W_val, name="$(name_prefix)_W")
    return EmbeddingLayer(W_var, embedding_dim, vocab_size)
end
function Conv1DLayer(kernel_width::Int, in_channels::Int, out_channels::Int, activation_graph_fn::Function; name_prefix::String="conv1d")
    K_val = glorot_uniform(kernel_width, in_channels, out_channels)
    b_val = zeros(Float32, out_channels)
    K_var = Variable(K_val, name="$(name_prefix)_K")
    b_var = Variable(b_val, name="$(name_prefix)_b")
    return Conv1DLayer(K_var, b_var, activation_graph_fn)
end

permute_dims(x_node::GraphNode, perm) = PermuteDimsNode(x_node, perm)
maxpool1d(x_node::GraphNode, pool_spec::Tuple, stride_spec::Tuple) = MaxPool1DNode(x_node, pool_spec, stride_spec)
flatten(x_node::GraphNode) = FlattenNode(x_node)

"""
Helper function to get all trainable Variable parameters from the model

"""

function get_trainable_params(model::CustomModel)
    params = Variable[]
    for layer_or_fn in model.layers
        if isa(layer_or_fn, DenseLayer)
            push!(params, layer_or_fn.W); push!(params, layer_or_fn.b)
        elseif isa(layer_or_fn, Conv1DLayer)
            push!(params, layer_or_fn.K); push!(params, layer_or_fn.b)
        elseif isa(layer_or_fn, EmbeddingLayer)
            push!(params, layer_or_fn.W)            
        end
    end
    return params
end

"""
Forward Pass for CustomModel
“Call” model like a function to run the forward graph

"""

function (model::CustomModel)(x_input_node::GraphNode)
    current_out_node = x_input_node
    for layer_or_fn in model.layers
        if isa(layer_or_fn, DenseLayer)
            z_node = (layer_or_fn.W * current_out_node) .+ layer_or_fn.b
            current_out_node = layer_or_fn.activation_graph_fn(z_node)
        elseif isa(layer_or_fn, EmbeddingLayer)
            current_out_node = EmbeddingNode(current_out_node, layer_or_fn.W)
        elseif isa(layer_or_fn, Conv1DLayer)
            conv_linear_part = Conv1DNode(current_out_node, layer_or_fn.K, layer_or_fn.b)
            current_out_node = layer_or_fn.activation_graph_fn(conv_linear_part)
        elseif isa(layer_or_fn, Function)
            current_out_node = layer_or_fn(current_out_node)
        else
            error("Unsupported layer or function type in CustomModel: $(typeof(layer_or_fn))")
        end
    end
    return current_out_node
end

"""
Update for Adam Optimizer

"""

function update!(opt::AdamOptimizer, model::CustomModel)
    opt.t += 1

    beta1_power_t = opt.beta1^opt.t
    beta2_power_t = opt.beta2^opt.t
    
    m_hat_coeff = 1.0f0 / (1.0f0 - beta1_power_t + eps(Float32))
    v_hat_coeff = 1.0f0 / (1.0f0 - beta2_power_t + eps(Float32))

    for param in get_trainable_params(model)
        grad = param.gradient

        if !haskey(opt.m, param)
            opt.m[param] = zeros(Float32, size(param.output))
            opt.v[param] = zeros(Float32, size(param.output))
        end

        m_p = opt.m[param] 
        v_p = opt.v[param] 

        @. m_p = opt.beta1 * m_p + (1.0f0 - opt.beta1) * grad
        @. v_p = opt.beta2 * v_p + (1.0f0 - opt.beta2) * grad^2.0f0
        @. param.output -= opt.lr * (m_p * m_hat_coeff) / (sqrt(v_p * v_hat_coeff) + opt.epsilon)
    end
end