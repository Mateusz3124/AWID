using Random

import Statistics: mean
accuracy(ŷ, y) = mean((ŷ .> 0.5) .== (y .> 0.5))

import LinearAlgebra
function startNetwork(X_train::Matrix, y_train::Vector, X_test::Matrix, y_test::Vector, chain::Chain, batch_size::Int64, epoch_num::Int64)

    val2 = size(X_train, 2)

    if (val2 % batch_size != 0)
        throw(ArgumentError("User input invalid: Data of size $val2 could not be divided into batches of size $batch_size"))
    end

    if (size(X_train, 2) != length(y_train))
        throw(ArgumentError("x train data size doesn't match y train size"))
    end

    if (size(X_test, 2) != length(y_test))
        throw(ArgumentError("x test data size doesn't match y test size"))
    end

    batchNum = Int(val2 / batch_size)

    data = init_data(X_train, y_train, batchNum, batch_size)
    shuffle!(data)
    
    x = Variable(data[1][1])
    y = Variable(data[1][2])

    xTest = Variable(X_test)
    yTest = Variable(y_test)

    (graph, ŷ, opt_values) = graph_build(x, y, chain)
    (graphTest, ŷ_test) = graph_build_from_opt_values(xTest, yTest, chain, opt_values)

    optimizer = Adam(opt_values)

    for epoch in 1:epoch_num
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
end