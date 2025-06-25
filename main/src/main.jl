module main
using JLD2
using net

# @load "../data/imdb_dataset_prepared.jld2" X_train y_train X_test y_test

# X_train = Matrix{Float32}(X_train)
# y_train = Vector{Float32}(vec(y_train))
# X_test = Matrix{Float32}(X_test)
# y_test = Vector{Float32}(vec(y_test))

# chain = Chain(Dense(size(X_train, 1), 32, relu), Dense(32, 1, σ), Classification(binarycrossentropy))

# startNetwork(X_train, y_train, X_test, y_test, chain, 64, 5)

@load "../data/imdb_dataset_prepared_embedings.jld2" X_train y_train X_test y_test embeddings vocab

X_train = Int32.(X_train)
y_train = Float32.(vec(y_train))
X_test = Int32.(X_test)
y_test = Float32.(vec(y_test))

chain = Chain(
    Embedding(embeddings),
    Convolution(3, size(embeddings, 1), 8, relu),
    MaxPool(8),
    Flatten(),
    Dense(128, 1, σ),
    Classification(binarycrossentropy)
)


startNetwork(X_train, y_train, X_test, y_test, chain, 64, 5)

end