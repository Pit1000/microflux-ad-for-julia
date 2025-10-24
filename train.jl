"""
Training Function

"""

function custom_model_predict(model::CustomModel, x_data::AbstractArray)
    x_input_node = Constant(x_data)
    output_node = model(x_input_node)
    order = topological_sort(output_node)
    return forward!(order)
end

function eval_accuracy_fn_custom(model::CustomModel, x_eval::AbstractArray, y_eval::AbstractArray)
    preds_raw = custom_model_predict(model, x_eval) 

    count_correct = 0
    len = length(preds_raw)
    if len == 0 return 0.0f0 end

    @simd for i in eachindex(preds_raw, y_eval)
        pred_bool = preds_raw[i] > 0.5f0
        true_bool = y_eval[i] > 0.5f0 
        count_correct += (pred_bool == true_bool)
    end
    return Float32(count_correct / len)
end

function eval_loss_fn_custom(model::CustomModel, x_eval::AbstractArray, y_eval::AbstractArray, num_elements_for_mean::Int)
    x_input_node = Constant(x_eval)
    y_target_node = Constant(y_eval) 
    num_elements_node = Constant(Float32(num_elements_for_mean))

    output_probs_node = model(x_input_node)
    loss_node = binarycrossentropy_graph_loss(output_probs_node, y_target_node, num_elements_node)

    order = topological_sort(loss_node)
    loss_val = forward!(order)
    return loss_val
end

function train_custom_model!(custom_model::CustomModel, training_config::TrainingConfig, X_train_data, y_train_data, X_test_data, y_test_data; is_seq_data::Bool=false)

    num_train_samples = size(X_train_data)[end]
    batch_size = training_config.batch_size
    num_batches_total = ceil(Int, num_train_samples / batch_size)

    println("Starting training...")
    for epoch in 1:training_config.epochs
        total_loss_epoch = 0.0f0
        shuffled_indices = randperm(num_train_samples)
        t_epoch = @elapsed begin
            for i in 1:num_batches_total
                start_idx = (i-1) * batch_size + 1
                end_idx = min(i * batch_size, num_train_samples)
                batch_indices_range = start_idx:end_idx
                current_batch_size = length(batch_indices_range)
                
                x_batch = selectdim(X_train_data, ndims(X_train_data), shuffled_indices[batch_indices_range])
                y_batch = selectdim(y_train_data, ndims(y_train_data), shuffled_indices[batch_indices_range])

                x_input_node = Constant(x_batch)
                y_target_node = Constant(y_batch)
                num_elements_in_batch_node = Constant(Float32(current_batch_size))

                output_probs_node = custom_model(x_input_node)
                loss_graph_node = binarycrossentropy_graph_loss(output_probs_node, y_target_node, num_elements_in_batch_node)

                order = topological_sort(loss_graph_node)
                current_batch_loss_val = forward!(order)
                backward!(order)
                update!(training_config.optimizer, custom_model)

                total_loss_epoch += current_batch_loss_val * current_batch_size
            end
        end

        train_loss = num_train_samples > 0 ? total_loss_epoch / num_train_samples : 0.0f0
        train_acc = eval_accuracy_fn_custom(custom_model, X_train_data, y_train_data)

        num_test_samples = size(X_test_data)[end]
        test_loss = num_test_samples > 0 ? eval_loss_fn_custom(custom_model, X_test_data, y_test_data, num_test_samples) : 0.0f0
        test_acc = num_test_samples > 0 ? eval_accuracy_fn_custom(custom_model, X_test_data, y_test_data) : 0.0f0

        @printf("Epoch: %d (%.2f s)\tTrain: (l: %.4f, a: %.4f)\tTest: (l: %.4f, a: %.4f)\n", epoch, t_epoch, train_loss, train_acc, test_loss, test_acc)
    end
    println("Training finished.")
end

function main_cnn()

    data_path = "C:\\Users\\Kapitan Nemo\\Documents\\CNN\\data\\imdb_dataset_prepared.jld2"
    local X_train, y_train, X_test, y_test, pretrained_embeddings, vocab_list

    println("Loading data from: $data_path")
    loaded_data = load(data_path)
    X_train = loaded_data["X_train"]
    y_train = loaded_data["y_train"]
    X_test = loaded_data["X_test"]
    y_test = loaded_data["y_test"]
    pretrained_embeddings = loaded_data["embeddings"]

    vocab_list = loaded_data["vocab"]
    println("Data loaded successfully.")
    
    vocab_size = size(pretrained_embeddings, 2)
    embedding_dim = size(pretrained_embeddings, 1)

    cnn_model = Any[
        EmbeddingLayer(vocab_size, embedding_dim, name_prefix="embed"),
        (x -> permute_dims(x, (2,1,3))), 
        Conv1DLayer(3, embedding_dim, 8, relu_graph, name_prefix="conv1"), #conv_kernel_width, embedding_dim, conv_out_channels   
        (x -> maxpool1d(x, (8,), (8,))), #maxpool_pool_width, maxpool_stride_width
        flatten,
        DenseLayer(128, 1, sigmoid_graph, name_prefix="dense_out") #flattened_dim 
    ]
    custom_cnn_model = CustomModel(cnn_model)

    adam_optimizer = AdamOptimizer(0.001f0)
    batch_size = min(64, size(X_train,2))
    training_cfg = TrainingConfig(
        5, # epochs
        batch_size, # batch_size
        adam_optimizer
        ) 

    println("\nTraining CNN Model:")
    train_custom_model!(custom_cnn_model, training_cfg, X_train, y_train, X_test, y_test)
    println("\nTraining CNN Model finished.")

    println("\nTesting prediction with a few IMDB test samples")
    num_samples_to_test = min(3, size(X_test, 2))
    if num_samples_to_test > 0
        for i in 1:num_samples_to_test
            single_sample_x = selectdim(X_test, ndims(X_test), i:i)
            true_label = y_test[1, i]

            x_node_single = Constant(single_sample_x)
            pred_node_single = custom_cnn_model(x_node_single)
            order_single = topological_sort(pred_node_single)
            prediction_single_arr = forward!(order_single)
            prediction_prob = prediction_single_arr[1,1]

            println("\nSample $i:")
            println("True Label: $true_label")
            println("Predicted Probability: $prediction_prob")
            println("Predicted Class (threshold 0.5f0): $(prediction_prob > 0.5f0 ? 1.0f0 : 0.0f0)")
        end
    end
end

function main_mlp()

    data_path = "C:\\Users\\Kapitan Nemo\\Documents\\μFlux_KM3\\data_mlp\\imdb_dataset_prepared.jld2"
    local X_train, y_train, X_test, y_test, pretrained_embeddings, vocab_list

    println("Loading data from: $data_path")
    loaded_data = load(data_path)
    X_train = loaded_data["X_train"]
    y_train = loaded_data["y_train"]
    X_test = loaded_data["X_test"]
    y_test = loaded_data["y_test"]
    println("Data loaded successfully.")

    # 1. Define Network Architecture Configuration
    mlp_model = [
        DenseLayer(size(X_train, 1), 32, relu), 
        DenseLayer(32, 1, sigmoid)
    ]

    # 2. Define Training Configuration with ADAM 
    adam_optimizer = AdamOptimizer(0.001f0)

    training_cfg = TrainingConfig(
        5, # epochs
        32, # batch_size
        adam_optimizer
    )

    # 3. Build custom model
    custom_model = CustomModel(mlp_model)

    # 4. Train
    train_custom_model!(custom_model, training_cfg, X_train, y_train, X_test, y_test)

end

#@time main_mlp()

#@time main_cnn() 