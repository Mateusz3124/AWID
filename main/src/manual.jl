using JLD2
using net
using Random

@load "../data/imdb_dataset_prepared_embedings.jld2" X_train y_train X_test y_test embeddings vocab

X_train = Int32.(X_train)
y_train = Float32.(vec(y_train))
X_test = Int32.(X_test)
y_test = Float32.(vec(y_test))

batchNum = Int(size(X_train, 2) / 64)

data = init_data(X_train, y_train, batchNum, 64)
shuffle!(data)

x = Variable(data[1][1])
y = Variable(data[1][2])

xTest = Variable(X_test)
yTest = Variable(y_test)

chain = Chain(
    Embedding(embeddings),
    Convolution(3, size(embeddings, 1), 8, relu),
    MaxPool(8),
    Flatten(),
    Dense(128, 1, σ),
    Classification(binarycrossentropy)
)

(graph, ŷ, opt_values) = graph_build(x, y, chain)
(graphTest, ŷ_test) = graph_build_from_opt_values(xTest, yTest, chain, opt_values)

optimizer = Adam(opt_values)

for epoch in 1:5
    total_loss = 0.0f0
    total_acc = 0.0f0

    t = @elapsed begin
        for (x_curr, y_curr) in data
            x.output .= x_curr
            y.output .= y_curr

            forward_pass!(graph)

            total_loss += graph[end].output[1]
            total_acc += accuracy(ŷ.output, y_curr)

            backward_pass!(graph)

            apply!(optimizer, opt_values)
        end
        forward_pass!(graphTest)

        test_loss = graphTest[end].output[1]
        test_acc = accuracy(ŷ_test.output, y_test)
        shuffle!(data)
    end

    println("Epoch: $epoch, time: $t, Loss: $(total_loss/batchNum), Accuracy: $(total_acc/batchNum), LossTest: $(test_loss), AccuracyTest: $(test_acc)")
end