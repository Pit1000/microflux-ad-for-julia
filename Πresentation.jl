push!(LOAD_PATH, pwd())

using JLD2
using μFlux

function train()
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

    embedding_layer = EmbeddingLayer(vocab_size, embedding_dim, name_prefix="embed")
    if size(embedding_layer.W.output) == size(pretrained_embeddings)
        embedding_layer.W.output .= pretrained_embeddings 
    end

    cnn_model = Any[
        embedding_layer,
        (x -> permute_dims(x, (2,1,3))), 
        Conv1DLayer(3, embedding_dim, 8, relu, name_prefix="conv1"), #kernel_width, embedding_dim, out_channels   
        (x -> maxpool1d(x, (8,), (8,))), #pool_width, stride_width
        flatten,
        DenseLayer(128, 1, sigmoid, name_prefix="dense_out") #flattened_dim 
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
end

@time train()

# Training CNN Model:
# Starting training...
# Epoch: 1 (28.78 s)      Train: (l: 0.5591, a: 0.8344)   Test: (l: 0.4063, a: 0.8173)
# Epoch: 2 (13.26 s)      Train: (l: 0.3454, a: 0.8840)   Test: (l: 0.3378, a: 0.8514)
# Epoch: 3 (13.25 s)      Train: (l: 0.2619, a: 0.9290)   Test: (l: 0.3113, a: 0.8694)
# Epoch: 4 (13.30 s)      Train: (l: 0.2021, a: 0.9518)   Test: (l: 0.3155, a: 0.8727)
# Epoch: 5 (13.30 s)      Train: (l: 0.1530, a: 0.9704)   Test: (l: 0.3345, a: 0.8698)
# Training finished.

# Training CNN Model finished.
# 81.899560 seconds (197.91 M allocations: 110.216 GiB, 8.42% gc time, 17.64% compilation time: <1% of which was recompilation)