"""
Constants for activation and loss functions

"""

ZERO_F32_NODE = Constant(0.0f0)
ONE_F32_NODE = Constant(1.0f0)
BCE_EPS_F32_NODE = Constant(eps(Float32))

function glorot_uniform(dims_tuple::Tuple)
    fan_in, fan_out = 0, 0
    if length(dims_tuple) == 1
        fan_in = dims_tuple[1]; fan_out = 1
    elseif length(dims_tuple) == 2
        fan_out, fan_in = dims_tuple[1], dims_tuple[2]
    elseif length(dims_tuple) >= 3
        receptive_field_size = prod(dims_tuple[1:end-2])
        c_in = dims_tuple[end-1]; c_out = dims_tuple[end]
        fan_in = c_in * receptive_field_size; fan_out = c_out * receptive_field_size
    else error("Cannot initialize empty dimensions.") end
    limit = sqrt(6.0f0 / (fan_in + fan_out))
    return (rand(Float32, dims_tuple...) .* 2.0f0 .- 1.0f0) .* limit
end
glorot_uniform(dims...) = glorot_uniform(dims)

"""
Graph Node representing activation fun

"""

relu(x_node::GraphNode) = max.(x_node, ZERO_F32_NODE)
sigmoid(x_node::GraphNode) = σ(x_node)
identity_graph(x_node::GraphNode) = x_node

"""
Mean Binary Cross Entropy for GraphNode

"""

function binarycrossentropy_graph_loss(p_node::GraphNode, y_node::GraphNode, N_elements_node::GraphNode)
    term1 = y_node .* log(p_node .+ BCE_EPS_F32_NODE)
    term2 = (ONE_F32_NODE .- y_node) .* log(ONE_F32_NODE .- p_node .+ BCE_EPS_F32_NODE)
    sum_of_terms_elementwise = term1 .+ term2
    sum_of_losses = sum(-(sum_of_terms_elementwise))
    mean_loss = sum_of_losses ./ N_elements_node
    return mean_loss
end