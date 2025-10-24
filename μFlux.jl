"""
    μFlux

    A μ auto-differentiation and neural-network training framework

"""

module μFlux

using Printf
using LinearAlgebra
using Statistics
using Random
using JLD2

include("graph.jl")
include("autodiff.jl")
include("utils.jl")
include("layers.jl")
include("train.jl")

export GraphNode, Variable, Constant, ScalarOperator, BroadcastedOperator, 
    topological_sort, forward!, backward!,
    relu, sigmoid, σ, EmbeddingNode, Conv1DNode,
    DenseLayerConfig, DenseLayer, CustomModel, TrainingConfig,
    AdamOptimizer,
    train_custom_model!,
    EmbeddingLayer, Conv1DLayer, 
    flatten, permute_dims, maxpool1d
end