"""
Abstract Types
Define the fundamental building blocks of the computation graph.

GraphNode is the base abstract type for all elements in the computation graph (e.g. variables, constants, operators)

"""
abstract type GraphNode end

"""
Operator is a subtype of GraphNode representing any computational operation in Graph.

"""

abstract type Operator <: GraphNode end

"""
Constant node in the computation graph that holds a fixed value (output).

"""

struct Constant{T} <: GraphNode 
    output :: T
end

"""
Variable is node representing a model parameter or input that can change during training.
Stores its current value (output), its gradient, and a readable name.

"""

mutable struct Variable{T} <: GraphNode 
    output :: T
    gradient :: Union{Nothing, T} 
    name :: String
    Variable(output::T; name="?") where T = new{T}(output, nothing, name)
end

""" 
ScalarOperator is a subtype of Operator, Represents scalar operations (e.g. addition +, multiplication *).
Stores its current value (output), its gradient, input a readable name.

"""

mutable struct ScalarOperator{F, I<:Tuple{Vararg{GraphNode}}} <: Operator
    inputs :: I
    output :: Union{Nothing, Number} 
    gradient :: Union{Nothing, Number}
    name :: String
    ScalarOperator(fun::F, inputs::GraphNode...; name="?") where F = new{F, typeof(inputs)}(inputs, nothing, nothing, name)
end

"""
BroadcastedOperator is a subtype of Operator, Represents element-wise or broadcasting operations (e.g., addition .+, multiplication .*, sigmoid)
Stores its current value (output), its gradient, input a readable name.

"""

mutable struct BroadcastedOperator{F, I<:Tuple{Vararg{GraphNode}}} <: Operator
    inputs :: I
    output :: Union{Nothing, AbstractArray, Number} 
    gradient :: Union{Nothing, AbstractArray, Number}
    name :: String
    BroadcastedOperator(fun::F, inputs::GraphNode...; name="?") where F = new{F, typeof(inputs)}(inputs, nothing, nothing, name)
end


"""
EmbeddingNode represents an embedding lookup operation. It takes integer indices and an embedding matrix (weights) as input,
and outputs the corresponding embedding vectors. Caches input indices for the backward pass.

"""

mutable struct EmbeddingNode <: Operator
    inputs :: Tuple{GraphNode, GraphNode} 
    output :: Union{Nothing, AbstractArray}
    gradient :: Union{Nothing, AbstractArray}
    name :: String
    indices_cache :: Union{Nothing, AbstractArray{<:Integer}} 
    EmbeddingNode(indices_node::GraphNode, W_node::GraphNode; name="embeddinglookup") = new((indices_node, W_node), nothing, nothing, name, nothing)
end

"""
PermuteDimsNode represents a dimension permutation operation.
It reorders the dimensions of an input array according to a specified permutation.
Stores the permutation and its inverse for forward and backward passes.

"""

mutable struct PermuteDimsNode{P<:Tuple{Vararg{Int}}} <: Operator
    inputs :: Tuple{GraphNode} 
    output :: Union{Nothing, AbstractArray}
    gradient :: Union{Nothing, AbstractArray}
    name :: String
    perm :: P
    inv_perm :: P
    PermuteDimsNode(input_node::GraphNode, perm::P; name="permute") where {P<:Tuple{Vararg{Int}}} = new{P}((input_node,), nothing, nothing, name, perm, invperm(perm))
end


"""
Conv1DNode represents a 1D convolutional operation.
It takes an input tensor, a kernel tensor, and a bias vector as inputs.
Caches the `im2col` transformed input (`x_col_cache`) for efficient gradient computation.

"""

mutable struct Conv1DNode <: Operator
    inputs :: Tuple{GraphNode, GraphNode, GraphNode} 
    output :: Union{Nothing, AbstractArray}
    gradient :: Union{Nothing, AbstractArray}
    name :: String
    x_col_cache :: Union{Nothing, AbstractArray} 
    Conv1DNode(input_node::GraphNode, kernel_node::GraphNode, bias_node::GraphNode; name="conv1d") = new((input_node, kernel_node, bias_node), nothing, nothing, name, nothing)
end

"""
MaxPool1DNode represents a 1D max pooling operation.
It takes an input tensor and applies max pooling over 1D windows.
Stores pooling window specifications, argmax indices, and input shape for the backward pass.

"""

mutable struct MaxPool1DNode{PoolSpec<:Tuple{Int}} <: Operator
    inputs :: Tuple{GraphNode} 
    output :: Union{Nothing, AbstractArray}
    gradient :: Union{Nothing, AbstractArray}
    name :: String
    pool_spec :: PoolSpec 
    stride_spec :: PoolSpec 
    argmax_cache :: Union{Nothing, AbstractArray{<:Integer}} 
    input_shape_cache :: Union{Nothing, Tuple{Vararg{Int}}} 
    MaxPool1DNode(input_node::GraphNode, pool_spec::PS, stride_spec::PS; name="maxpool1d") where {PS<:Tuple{Int}} = new{PS}((input_node,), nothing, nothing, name, pool_spec, stride_spec, nothing, nothing)
end

"""
FlattenNode represents an operation that flattens a multi-dimensional input
into a 2D matrix, typically preserving the last dimension as the batch dimension.
Caches the original input shape for unflattening during the backward pass.

"""

mutable struct FlattenNode <: Operator
    inputs :: Tuple{GraphNode} 
    output :: Union{Nothing, AbstractArray}
    gradient :: Union{Nothing, AbstractArray}
    name :: String
    original_shape_cache :: Union{Nothing, Tuple{Vararg{Int}}} 
    FlattenNode(input_node::GraphNode; name="flatten") = new((input_node,), nothing, nothing, name, nothing)
end


"""
Overrides Julia's default print/display behavior to make graph nodes and operators more readable when printed.

"""

import Base: show, summary
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
show(io::IO, x::Constant) = print(io, "const ", summary(x.output)) # summary to handle arrays concisely
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name);
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ "); summary(io, x.gradient)
end
show(io::IO, x::EmbeddingNode) = print(io, "op ", x.name);
show(io::IO, x::PermuteDimsNode) = print(io, "op ", x.name, " ", x.perm);
show(io::IO, x::Conv1DNode) = print(io, "op ", x.name);
show(io::IO, x::MaxPool1DNode) = print(io, "op ", x.name, " pool", x.pool_spec, " stride", x.stride_spec);
show(io::IO, x::FlattenNode) = print(io, "op ", x.name);


"""
Function visit for graph traversal. Adds a node to the visited set and order

Overloaded for Operator types to recursively visit all their input nodes first—ensuring correct dependency ordering

"""

function visit(node::GraphNode, visited::Set{GraphNode}, order::Vector{GraphNode})
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end


function visit(node::Operator, visited::Set{GraphNode}, order::Vector{GraphNode})
    if node ∈ visited
    else
        push!(visited, node)
        for input_node in node.inputs
            visit(input_node, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

"""
Produces a list of nodes sorted in topological order for further forward computation

"""

function topological_sort(head::GraphNode)
    visited = Set{GraphNode}()
    order = Vector{GraphNode}()
    visit(head, visited, order)
    return order
end

"""
Reduces the input `x` by summing over specific dimensions, so that its shape matches `target_shape`. 

"""

function sum_to_shape(x::AbstractArray, target_shape::Tuple)
    if size(x) == target_shape
        return x
    end
    # Phase 1: Sum up the dimensions that are equal to 1 in target_shape and greater in x.
    dims_to_sum_phase1 = Int[]
    for d in 1:length(target_shape) # Iterate through the dimensions of the target shape.
        if target_shape[d] == 1 && size(x, d) > 1 # If the target dimension is 1, and w x is greater....
            push!(dims_to_sum_phase1, d) # ... add this dimension to the list for summation.
        end
    end

    g_intermediate = x
    if !isempty(dims_to_sum_phase1)
        # Sum along the dimensions identified in phase 1.
        g_intermediate = sum(g_intermediate, dims=Tuple(dims_to_sum_phase1))
    end

    # Phase 2: Summarize dimensions (those that exist in g_intermediate but not in target_shape)
    dims_to_sum_phase2 = Int[]
    if ndims(g_intermediate) > length(target_shape) # If g_intermediate has more dimensions than target_shape....
        for d in (length(target_shape) + 1):ndims(g_intermediate) # ...iterate over these redundant dimensions.
            push!(dims_to_sum_phase2, d) # Add them to the list for summing.
        end
    end

    if !isempty(dims_to_sum_phase2)
        # Sum along excess dimensions.
        g_intermediate = sum(g_intermediate, dims=Tuple(dims_to_sum_phase2))
    end

    # if the shape still does not match, hit it with reshape.
    if size(g_intermediate) == target_shape
        return g_intermediate
    else
        try
            return reshape(g_intermediate, target_shape)
        catch e
            error("sum_to_shape: Final reshape failed from $(size(g_intermediate)) (after summing dims $(dims_to_sum_phase1) and $(dims_to_sum_phase2)) to $target_shape. Original x: $(size(x)). Error: $e")
        end
    end
end
sum_to_shape(x::AbstractArray, target_shape::Tuple{}) = sum(x)
sum_to_shape(x::Number, target_shape::Tuple{}) = x
function sum_to_shape(x::Number, target_shape::Tuple)
    if isempty(target_shape)
        return x
    else
        error("sum_to_shape called with scalar input x and non-empty target_shape $target_shape. This is unusual.")
    end
end