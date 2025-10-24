"""
Reset gradient

# Before each forward pass and calculation of new gradients,
# old gradients is reset to zero (set to `nothing`).

"""

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

"""
Compute forward output for a node

"""

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)

"""
Function forward performs a forward pass through the graph in topologically sorted order.
Resets the state of each node, computes outputs for all operator nodes,
and returns the final output from the last node.

"""

function forward!(order::Vector{GraphNode})
    for node in order
        reset!(node) 
        compute!(node) 
    end
    return last(order).output
end

"""
Update gradients of nodes 

"""

update!(node::Constant, gradient) = nothing

"""

Updates the gradient of the `node::Variable` variable.
If the variable's gradient was `nothing`, sets it to `gradient_val`.
Otherwise, adds `gradient_val` to the existing gradient.
Supports both arrays (element-by-element addition) and numbers.

"""

@inline function update!(node::Variable{T}, gradient_val::T) where T 
    if node.gradient === nothing
        node.gradient = gradient_val
    else
        if T <: AbstractArray 
            node.gradient .+= gradient_val
        else 
            node.gradient += gradient_val
        end
    end
end

"""
Updates the gradient of the output of the `node::Operator` operator.
As for `Variable`, accumulates gradients.

"""

function update!(node::Operator, gradient_val::Union{AbstractArray, Number})
    current_op_grad = node.gradient
    if node.gradient === nothing
        node.gradient = gradient_val
    else
        if isa(node.gradient, AbstractArray) && isa(gradient_val, AbstractArray)
            if size(node.gradient) == size(gradient_val)
                node.gradient .+= gradient_val
            else
                error("Gradient size mismatch for operator $(node.name): $(size(current_op_grad)) vs $(size(gradient_val))")
            end
        elseif isa(node.gradient, AbstractArray) && isa(gradient_val, Number)
            node.gradient .+= gradient_val     
        elseif isa(node.gradient, Number) && isa(gradient_val, Number)
            node.gradient = node.gradient + gradient_val
        elseif isa(node.gradient, Number) && isa(gradient_val, AbstractArray)
            @warn "Operator $(node.name) gradient was scalar, now receiving array. Promoting."
            node.gradient = fill(node.gradient, size(gradient_val)) .+ gradient_val
        else
            error("Incompatible gradient types for operator $(node.name): $(typeof(current_op_grad)) and $(typeof(gradient_val))")
        end
    end
end

"""
Perform the reverse-mode automatic-differentiation pass over a computational graph 

"""

function backward!(order::Vector{GraphNode}; seed::Real=1.0f0)
    result_node = last(order) # The last node in topological order (the main result of the graph).
    if result_node.output === nothing
        error("Cannot seed gradient: graph output is Nothing. Run forward pass first.")
    end

    # Set the gradient for the resulting node.
    if isa(result_node.output, AbstractArray)
        el_type = eltype(result_node.output) 
        result_node.gradient = fill(convert(el_type, seed), size(result_node.output))
    elseif isa(result_node.output, Number)
        result_node.gradient = convert(typeof(result_node.output), seed)
    else
        error("Cannot seed gradient for node with output type $(typeof(result_node.output))")
    end

    # Propagate gradients backward through the graph
    for node in reverse(order)
        if isa(node, Operator)
            backward!(node)
        end
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end

@inline function backward!(node::Operator) 

    (node.gradient === nothing) && return nothing

    input_outputs = map(input_node -> input_node.output, node.inputs) 
    gradients_for_inputs::Tuple = backward(node, input_outputs..., node.gradient)

    # Loop iterating over pairs (input node, corresponding gradient)
    @inbounds for i in eachindex(node.inputs, gradients_for_inputs) 
        input_node = node.inputs[i] # Current operator input node
        grad_val = gradients_for_inputs[i] # Gradient calculated for this input_node.
        # If `grad_val` is not nothing, it means that there is a meaningful gradient to pass to input_node.
        if grad_val !== nothing  
            update!(input_node, grad_val)
        end
    end
    return nothing
end



# EmbeddingNode
function forward(node::EmbeddingNode, indices_val::AbstractArray{<:Integer}, W_val::AbstractArray)
    node.indices_cache = indices_val # Cache input indices for backward pass
    embedding_dim, vocab_size = size(W_val)
    indices_ndims = ndims(indices_val)
    local indices_2d::AbstractMatrix{<:Integer}

    if indices_ndims == 1 #If indices_val is 1D
        # seq_len = length(indices_val)
        # batch_size = 1
        indices_2d = reshape(indices_val, :, 1) # Reshape to (seq_len, 1)
    elseif indices_ndims == 2 # If indices_val is already 2D
        # seq_len, batch_size = size(indices_val)
        indices_2d = indices_val
    else
        error("Embedding input indices must be 1D or 2D. Got $(indices_ndims)D.")
    end
    seq_len, batch_size = size(indices_2d)

    output_arr = similar(W_val, embedding_dim, seq_len, batch_size)

    # Iterate through each item in the batch and each token in the sequence for that item
    @inbounds for k_batch in 1:batch_size
        for s_seq in 1:seq_len
            vocab_idx = indices_2d[s_seq, k_batch]
            # if !(1 <= vocab_idx <= vocab_size)
            #     error("Vocab index $vocab_idx out of bounds [1, $vocab_size] at seq_pos $s_seq, batch_item $k_batch.")
            # end
            dest_view = @view output_arr[:, s_seq, k_batch]
            src_view = @view W_val[:, vocab_idx]
            copyto!(dest_view, src_view)
        end
    end
    return output_arr
end


function backward(node::EmbeddingNode, indices_val::AbstractArray{<:Integer}, W_val::AbstractArray, g_output_val::AbstractArray)
    # Use cached indices from forward pass
    indices_cached = node.indices_cache
    local indices_2d::AbstractMatrix{<:Integer}
    if ndims(indices_cached) == 1 # If indices_cached is 1D
        indices_2d = reshape(indices_cached, :, 1)  # Reshape it to a 2D matrix with one column
    else
        indices_2d = indices_cached  #If indices_cached is already 2D, use it as is.
    end

    # Extract sequence length and batch size from the (for sure) 2D indices matrix.
    seq_len, batch_size = size(indices_2d)

    # Dimensions of the embedding weight matrix W_val.
    # embedding_dim: The size of each embedding vector.
    # vocab_size: The total number of unique tokens in the vocabulary.
    embedding_dim, vocab_size = size(W_val)
    # Initialize the gradient matrix for W (grad_W) with zeros.
    grad_W = zeros(eltype(W_val), embedding_dim, vocab_size) 

    # Get the dimensions of the incoming gradient g_output_val.
    # This gradient comes from the layer that followed the embedding layer.
    g_embedding_dim, g_seq_len, g_batch_size = size(g_output_val)
    #@assert g_embedding_dim == embedding_dim && g_seq_len == seq_len && g_batch_size == batch_size "Gradient shape mismatch in Embedding backward"

    # Iterate through each token in each batch item to distribute the gradients.
    @inbounds for k_batch in 1:g_batch_size # Loop over each item in the batch.
        for s_seq in 1:g_seq_len # Loop over each token in the current sequence.

            # Get the specific vocabulary index (word ID) for the current token (s_seq) in the current batch item (k_batch).
            vocab_idx = indices_2d[s_seq, k_batch]

            # Create a view to the column in `grad_W`
            dest_view_grad_W = @view grad_W[:, vocab_idx]
            # view to the incoming gradient vector
            src_view_g_output = @view g_output_val[:, s_seq, k_batch]
            # Accumulate the gradient: Add the incoming gradient to corresponding col in grad_W
            dest_view_grad_W .+= src_view_g_output
        end
    end
    return tuple(nothing, grad_W) 
end


# PermuteDimsNode
# Directly uses Julia's efficient permutedims function to perform data reordering.
@inline forward(node::PermuteDimsNode, x_val::AbstractArray) = permutedims(x_val, node.perm)
# Applies node.inv_perm (the inverse of the original permutation) to the incoming gradient g_val.
@inline backward(node::PermuteDimsNode, x_val::AbstractArray, g_val::AbstractArray) = tuple(permutedims(g_val, node.inv_perm))

"""
# It's a preprocessing step to transform input data into a matrix format
# for 1D convolutions to be performed efficiently using matrix multiplication.

"""

function im2col1d(x::AbstractArray{T,3}, KW::Int) where T
    W_in, C_in, N = size(x)
    # output width
    W_out = W_in - KW + 1
    # output matrix Xc
    Xc = Array{T}(undef, KW*C_in, W_out*N)

    current_col_idx_Xc = 1
    @inbounds for n_batch in 1:N
        for w_start_in_x in 1:W_out
            dest_row_start_in_Xc_col = 1
            for c_channel in 1:C_in
                # Base index for writing to the current column of Xc.
                # Adjusts for 0-based thinking in loop vs 1-based Julia indexing.
                idx_Xc_base = dest_row_start_in_Xc_col - 1
                # Base index for reading from the input x.
                idx_x_base  = w_start_in_x - 1
                # This loop copies KW elements from the current patch - for the current channel
                # into the appropriate rows of the current column in Xc.
                @simd for k_idx in 1:KW # Iterate over the elements within the kernel window.
                    Xc[idx_Xc_base + k_idx, current_col_idx_Xc] = x[idx_x_base + k_idx, c_channel, n_batch]
                end
                # Move the starting row for the next channel's data in Xc's current column.
                dest_row_start_in_Xc_col += KW
            end
            # Move to the next column in X
            current_col_idx_Xc += 1
        end
    end
    # Return the transformed matrix Xc. Each column of Xc now represents a "flattened"
    return Xc
end

"""
It's the inverse operation of `im2col`. In the context of backpropagation for convolutions.
Is used to map gradients from the column-matrix format back to the original input's shape.

"""

function col2im1d!(dx::AbstractArray{T,3}, dXcol::AbstractArray{T,2}, KW::Int) where T
    W_in, C_in, N = size(dx)
    W_out = W_in - KW + 1 

    # Initialize the output gradient array dx to zeros.
    fill!(dx, zero(T))

    # tracks which column of the input gradient matrix dXcol we are currently reading from.
    current_col_idx_dXcol = 1
    @inbounds for n_batch in 1:N # Iterate over each item in the batch.
        for w_start_in_dx in 1:W_out # w_start_in_dx is the starting 1-based index in the dx array's width dimension
            # src_row_start_in_dXcol_col tracks the starting row within the current column of dXcol
            # from where data for the current channel of the patch will be read.
            src_row_start_in_dXcol_col = 1
            for c_channel in 1:C_in # Iterate over each channel.
                dest_dx_base = w_start_in_dx - 1 
                src_dXcol_base = src_row_start_in_dXcol_col - 1
                # Loop takes KW elements from the current column of dXcol (representing
                # a single channel's contribution within a patch) and adds them to the corresponding locations in dx.
                @simd for k_idx in 1:KW # Iterate over the elements within the kernel window.
                    dx[dest_dx_base + k_idx, c_channel, n_batch] += dXcol[src_dXcol_base + k_idx, current_col_idx_dXcol]
                end
                # Move the starting row for reading the next channel's data from dXcol's current column.
                src_row_start_in_dXcol_col += KW
            end
            current_col_idx_dXcol += 1
        end
    end
    return dx # Return the modified dx array, which now contains the accumulated gradients, mapped back to the original input's spatial layout.
end

"""
Implementation forward pass for a 1D convolution operation.
Realises transform the convolution into a matrix multiplication.

"""

function conv1d_forward(node::Conv1DNode, x::AbstractArray{T,3}, K::AbstractArray{T,3}, b::AbstractVector{T}) where T
    W_in, C_in, N = size(x)
    KW, _, C_out = size(K) # Kernel is (KW, C_in, C_out)
    W_out = W_in - KW + 1  # Calculate the output

    # Transform input x using im2col
    Xc = im2col1d(x, KW) # Xc will have dimensions (KW*C_in, W_out*N)

    # Cache the `Xc` matrix in the `node` object for the backward pass,
    # as it avoids recomputing `im2col` during backpropagation.
    node.x_col_cache = Xc

    Km = reshape(K, KW*C_in, C_out) # Km will have dimensions (KW*C_in, C_out).

    # Pre-allocate the output matrix for `mul!`.
    Ymat_alloc = Array{T}(undef, C_out, W_out*N)
    # turns out to be faster than looping for convolution probable due to optimized BLAS routines.
    mul!(Ymat_alloc, transpose(Km), Xc)

    # Add bias using broadcasting
    Ymat_alloc .+= b

    # Reshape Ymat to (C_out, W_out, N) then permute to (W_out, C_out, N)
    Y = reshape(Ymat_alloc, C_out, W_out, N)
    return permutedims(Y, (2,1,3))
end


"""
Implementation the backward pass for a 1D convolution

"""

function conv1d_backward( node::Conv1DNode, g_y_val::AbstractArray{T,3}, x_val::AbstractArray{T,3}, K_val::AbstractArray{T,3}, b_val::AbstractVector{T}) where {T} 

    # Unpack dimensions from inputs.
    W_in, C_in, N = size(x_val)
    KW, C_in_K, C_out_K = size(K_val) # C_in_K should be C_in, C_out_K should be C_out
    W_out, C_out_gy, N_gy = size(g_y_val) # C_out_gy should be C_out, N_gy should be N

    g_y_perm = permutedims(g_y_val, (2, 1, 3))
    # Reshape the permuted gradient into a 2D
    dYmat = reshape(g_y_perm, C_out_gy, W_out * N_gy)

    g_b = vec(sum(dYmat, dims=2))

    # Retrieve Cached Xc 
    Xc = node.x_col_cache 

    # Pre-allocate the output matrix for `mul!`.
    g_K_mat_alloc = Array{T}(undef, KW * C_in, C_out_K)
    # Calculate Gradient (dL/dK)
    mul!(g_K_mat_alloc, Xc, transpose(dYmat))
    # Reshape g_K_mat_alloc back to the original 3D kernel shape (KW, C_in, C_out).
    g_K = reshape(g_K_mat_alloc, KW, C_in, C_out_K)

    # Prepare Kernel K_val for Matrix Operations (as Km) 
    Km_val_reshaped = reshape(K_val, KW * C_in, C_out_K)

    g_Xc_alloc = Array{T}(undef, KW * C_in, W_out * N_gy)
    #Calculate Gradient (dL/dXc)
    mul!(g_Xc_alloc, Km_val_reshaped, dYmat)

    g_x = similar(x_val, T) # `similar` preserves shape and sets eltype to T
    # Calculate Gradient (dL/dx)
    col2im1d!(g_x, g_Xc_alloc, KW)

    return g_x, g_K, g_b # Return the computed gradients.
end

@inline forward(node::Conv1DNode, x_val, K_val, b_val) = conv1d_forward(node, x_val, K_val, b_val)
function backward(node::Conv1DNode, x_val, K_val, b_val, g_y_val)
    g_x, g_K, g_b = conv1d_backward(node, g_y_val, x_val, K_val, b_val)
    return tuple(g_x, sum_to_shape(g_K, size(K_val)), sum_to_shape(g_b, size(b_val)))
end

"""
Implementation of forward pass for 1D max pooling.

"""

function maxpool1d_forward(x::AbstractArray{T,3}, pool_width::Int, stride_val::Int) where T
    W_in, C_channels, N_batch = size(x)
    W_out = fld(W_in - pool_width, stride_val) + 1

    # Pre-allocate the output array `Y` for the pooled values.
    Y = Array{T}(undef, W_out, C_channels, N_batch)
    # Pre-allocate the array `argmax_indices_abs` to store the abs indices (for backward gradients route)
    argmax_indices_abs = Array{Int}(undef, W_out, C_channels, N_batch) 

    @inbounds for n_idx in 1:N_batch # Iterate over each item in the batch
        @simd for c_idx in 1:C_channels # Iterate over each channel.
            for w_out_idx in 1:W_out # Iterate over each position in the output's width dimension.
                w_in_start = (w_out_idx-1)*stride_val + 1
                current_max_val = typemin(T)
                current_argmax_local_offset = 0 
                for k_pool_offset in 0:pool_width-1 # Iterate 0 to pool_width-1 for local offset
                    # Calculate the absolute index in `x`'s width dimension for the current element in the window.
                    val_idx_in_x = w_in_start + k_pool_offset
                    # Get the value from the input array.
                    val = x[val_idx_in_x, c_idx, n_idx]
                    if val > current_max_val # If the current value `val` is greater than `current_max_val`, update.
                        current_max_val = val
                        current_argmax_local_offset = k_pool_offset # Store the local offset of this new max.
                    end
                end
                # Store the maximum value found for this window in the output array `Y`.
                Y[w_out_idx, c_idx, n_idx] = current_max_val
                # Store the absolute index (in `x`) of this maximum value.
                argmax_indices_abs[w_out_idx, c_idx, n_idx] = w_in_start + current_argmax_local_offset
            end
        end
    end
    return Y, argmax_indices_abs
end

"""
Implementation of backward pass for 1D max pooling.

"""

function maxpool1d_backward(g_y::AbstractArray, argmax_indices::AbstractArray{Int}, x_input_shape::Tuple{Vararg{Int}}, pool_width::Int, stride_width::Int)
    g_x = zeros(eltype(g_y), x_input_shape) # Initialize the gradient array with zeros.
    W_out, C_gy, N_gy = size(g_y) # g_y is (W_out, C, N)

    @inbounds for n_batch in 1:N_gy # Iterate over each item in the batch.
        @simd for c_channel in 1:C_gy # Iterate over each channel.
            for w_out in 1:W_out # Iterate over each position in the pooled output's width dimension.
                # This index was stored in argmax_indices during the forward pass.
                original_w_idx = argmax_indices[w_out, c_channel, n_batch] 
                # Gradient routing (*)
                g_x[original_w_idx, c_channel, n_batch] += g_y[w_out, c_channel, n_batch]
            end
        end
    end
    return g_x
end

function forward(node::MaxPool1DNode, x_val::AbstractArray)
    pool_width = node.pool_spec[1]
    stride_width = node.stride_spec[1]
    y_val, argmax_idx = maxpool1d_forward(x_val, pool_width, stride_width)
    node.argmax_cache = argmax_idx
    node.input_shape_cache = size(x_val)
    return y_val
end

function backward(node::MaxPool1DNode, x_val::AbstractArray, g_y_val::AbstractArray)
    pool_width = node.pool_spec[1]
    stride_width = node.stride_spec[1]
    g_x = maxpool1d_backward(g_y_val, node.argmax_cache::AbstractArray{Int}, node.input_shape_cache::Tuple{Vararg{Int}}, pool_width, stride_width)
    return tuple(g_x)
end

"""
Implementation of forward pass for Flatten.

"""

function forward(node::FlattenNode, x_val::AbstractArray)
    original_shape = size(x_val)
    # Cache the original shape in the node.
    node.original_shape_cache = original_shape
    #if isempty(original_shape) error("Cannot flatten scalar or empty array.") end
    
    # Keep last dimension as batch_size, flatten all preceding dimensions
    if length(original_shape) == 1 # Single dimension input (Features, 1) 
        num_features = original_shape[1]
        batch_size = 1
        return reshape(x_val, num_features, batch_size)
    elseif length(original_shape) > 1 # If the input `x_val` has 2 or more dimensions.
        batch_size = original_shape[end]
        num_features = div(length(x_val), batch_size)
        return reshape(x_val, num_features, batch_size)
    else
        error("FlattenNode: Unexpected input shape $(original_shape)")
    end
end

"""
Implementation of backward pass for Flatten.

"""

function backward(node::FlattenNode, x_val::AbstractArray, g_y_val::AbstractArray)
    return tuple(reshape(g_y_val, node.original_shape_cache::Tuple{Vararg{Int}}))
end

"""
Base Operations Overloads 

"""

import Base: ^, sin, *, -, +, sum, /, max, exp, log

"""
Overload for power.

"""

^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

"""
Overload for sin.

"""

sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

"""
Overload for element-wise addition with broadcasting.

"""

Base.Broadcast.broadcasted(::typeof(+), x_node::GraphNode, y_node::GraphNode) = BroadcastedOperator(+, x_node, y_node, name=".+")
@inline forward(::BroadcastedOperator{typeof(+)}, x_val, y_val) = return x_val .+ y_val
function backward(::BroadcastedOperator{typeof(+)}, x_val, y_val, g_val)
    return tuple(sum_to_shape(g_val, size(x_val)), sum_to_shape(g_val, size(y_val)))
end

"""
Overload for matrix multiplication (or similar matrix-vector product, scalar-matrix product).

"""

*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x, name="*") # mul! implies matrix multiplication
@inline forward(::BroadcastedOperator{typeof(mul!)}, A_val, x_val) = return A_val * x_val # Standard * dispatch
function backward(::BroadcastedOperator{typeof(mul!)}, A_val, x_val, g_val)
    # Matrix multiplication or scalar-array products.
    # element-wise, use .*
    grad_A, grad_x = if ndims(A_val) >= 2 && ndims(x_val) >= 1 && size(A_val, 2) == size(x_val, 1) # Matrix multiplication A*x or A*X
        g_val * transpose(x_val), transpose(A_val) * g_val
    elseif isa(A_val, Number) && isa(x_val, AbstractArray) # Scalar * Array
        sum(g_val .* x_val), A_val .* g_val # grad_A is sum, grad_x is elementwise product with A_val
    elseif isa(A_val, AbstractArray) && isa(x_val, Number) # Array * Scalar
        g_val .* x_val, sum(g_val .* A_val) # grad_A is elementwise, grad_x is sum
    elseif isa(A_val, Number) && isa(x_val, Number) # Scalar * Scalar
        g_val * x_val, g_val * A_val
    end
    return tuple(sum_to_shape(grad_A, size(A_val)), sum_to_shape(grad_x, size(x_val)))
end

"""
Overload for element-wise multiplication with broadcasting.

"""

Base.Broadcast.broadcasted(::typeof(*), x_node::GraphNode, y_node::GraphNode) = BroadcastedOperator(*, x_node, y_node, name=".*")
@inline forward(::BroadcastedOperator{typeof(*)}, x_val, y_val) = return x_val .* y_val
function backward(::BroadcastedOperator{typeof(*)}, x_val, y_val, g_val)
    return tuple(sum_to_shape(g_val .* y_val, size(x_val)), sum_to_shape(g_val .* x_val, size(y_val)))
end

"""
Overload the unary minus

"""

negate_fn(x) = -x
-(x_node::GraphNode) = BroadcastedOperator(negate_fn, x_node, name="negate")
@inline forward(::BroadcastedOperator{typeof(negate_fn)}, x_val) = return -x_val
@inline backward(::BroadcastedOperator{typeof(negate_fn)}, x_val, g_val) = tuple(sum_to_shape(-g_val, size(x_val)))

"""
Overload for the element-wise subtraction

"""

Base.Broadcast.broadcasted(::typeof(-), x_node::GraphNode, y_node::GraphNode) = BroadcastedOperator(-, x_node, y_node, name=".-")
@inline forward(::BroadcastedOperator{typeof(-)}, x_val, y_val) = return x_val .- y_val
function backward(::BroadcastedOperator{typeof(-)}, x_val, y_val, g_val)
    return tuple(sum_to_shape(g_val, size(x_val)), sum_to_shape(-g_val, size(y_val)))
end

sum_op_fn(x) = Base.sum(x) 

"""
Overload the sum function

"""
sum(x_node::GraphNode; dims=nothing) = BroadcastedOperator(sum_op_fn, x_node, name="sum")
@inline forward(::BroadcastedOperator{typeof(sum_op_fn)}, x_val) = return Base.sum(x_val)
function backward(::BroadcastedOperator{typeof(sum_op_fn)}, x_val::AbstractArray, g_val::Real)
    return tuple(fill(convert(eltype(x_val), g_val), size(x_val)))
end
function backward(::BroadcastedOperator{typeof(sum_op_fn)}, x_val::Real, g_val::Real)
    return tuple(convert(typeof(x_val), g_val))
end

"""
Overload for the element-wise division

"""

Base.Broadcast.broadcasted(::typeof(/), x_node::GraphNode, y_node::GraphNode) = BroadcastedOperator(/, x_node, y_node, name="./")
@inline forward(::BroadcastedOperator{typeof(/)}, x_val, y_val) = return x_val ./ y_val
function backward(node::BroadcastedOperator{typeof(/)}, x_val, y_val, g_val)
    eps_ = eps(eltype(y_val))
    safe_y_val = y_val .+ eps_ .* (y_val .== zero(eltype(y_val)))
    grad_x_raw = g_val ./ safe_y_val
    grad_y_raw = g_val .* (-x_val ./ (safe_y_val .^ 2)) 
    return tuple(sum_to_shape(grad_x_raw, size(x_val)), sum_to_shape(grad_y_raw, size(y_val)))
end

"""
Overload for the element-wise max

"""

Base.Broadcast.broadcasted(::typeof(max), x_node::GraphNode, y_node::GraphNode) = BroadcastedOperator(max, x_node, y_node, name="max")
@inline forward(::BroadcastedOperator{typeof(max)}, x_val, y_val) = return max.(x_val, y_val)
function backward(::BroadcastedOperator{typeof(max)}, x_val, y_val, g_val)
    T = eltype(x_val) 
    mask_x_ge_y = T.(x_val .>= y_val) 
    dx_raw = g_val .* mask_x_ge_y
    dy_raw = g_val .* T.(y_val .> x_val) 
    return tuple(sum_to_shape(dx_raw, size(x_val)), sum_to_shape(dy_raw, size(y_val)))
end

"""
Overload for the element-wise σ function

"""

σ_fn(x) = 1.0f0 ./ (1.0f0 .+ exp.(-x))
σ(x_node::GraphNode) = BroadcastedOperator(σ_fn, x_node, name="σ")
@inline forward(::BroadcastedOperator{typeof(σ_fn)}, x_val) = return σ_fn(x_val) 
@inline backward(node::BroadcastedOperator{typeof(σ_fn)}, x_val, g_val) = tuple(sum_to_shape(g_val .* node.output .* (1.0f0 .- node.output), size(x_val)))

"""
Overload for the element-wise power

"""

Base.Broadcast.broadcasted(::typeof(^), x_node::GraphNode, y_node::GraphNode) = BroadcastedOperator(^, x_node, y_node, name=".^")
forward(::BroadcastedOperator{typeof(^)}, x_val, y_val) = return x_val .^ y_val
function backward(node::BroadcastedOperator{typeof(^)}, x_val, y_val, g_val)
    P_T = eltype(g_val)
    eps_val = eps(P_T)
    raw_grad_x = zero(x_val)
    if isa(y_val, Number) 
        raw_grad_x = g_val .* y_val .* (x_val .^ (y_val - one(P_T)))
    else 
        raw_grad_x = g_val .* y_val .* (x_val .^ (y_val .- one(P_T)))
    end
    raw_grad_y = g_val .* log.(abs.(x_val) .+ eps_val) .* node.output
    return tuple(sum_to_shape(raw_grad_x, size(x_val)), sum_to_shape(raw_grad_y, size(y_val)))
end

"""
Overload for the element-wise exp

"""

Base.Broadcast.broadcasted(::typeof(exp), x_node::GraphNode) = BroadcastedOperator(exp, x_node, name="exp")
@inline forward(::BroadcastedOperator{typeof(exp)}, x_val) = return exp.(x_val)
@inline backward(node::BroadcastedOperator{typeof(exp)}, x_val, g_val) = tuple(sum_to_shape(g_val .* node.output, size(x_val)))

"""
Overload for the element-wise log

"""

Base.Broadcast.broadcasted(::typeof(log), x_node::GraphNode) = BroadcastedOperator(log, x_node, name="log")
@inline forward(::BroadcastedOperator{typeof(log)}, x_val) = return log.(x_val .+ eps(eltype(x_val)))
@inline backward(::BroadcastedOperator{typeof(log)}, x_val, g_val) = tuple(sum_to_shape(g_val ./ (x_val .+ eps(eltype(x_val))), size(x_val)))
Base.log(x_node::GraphNode) = Base.Broadcast.broadcasted(log, x_node)