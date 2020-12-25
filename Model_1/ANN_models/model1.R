recall = c()
accuracy = c()
precision = c()
auc = c()
loss = c()

for(i in 1:10){
    test_index = which(folds == i, arr.ind = T)
    
    test_x = as.matrix(rand_data[test_index, 2:ncol(rand_data)])
    test_y = as.matrix(rand_data[test_index, 1])
    
    train_x = as.matrix(rand_data[-test_index, 2:ncol(rand_data)])
    train_y = as.matrix(rand_data[-test_index, 1])
    
    model = keras_model_sequential()
    
    model %>%
        layer_dense(units = 100, activation = 'relu', input_shape = c(ncol(train_x))) %>%
        layer_dense(units = 50, activation = 'relu') %>%
        layer_dense(units = 1, activation = 'sigmoid')
    
    model %>%
        compile(loss = 'binary_crossentropy',
                optimizer = optimizer_rmsprop(lr = lr),
                metrics = c('accuracy', tf$keras$metrics$AUC(), tf$keras$metrics$Precision(), tf$keras$metrics$Recall()))
    
    mymodel = model %>%
        fit(train_x,
            train_y,
            epochs = ep,
            batch_size = 32,
            validation_split = 0.2
        ) 
    
    eval = model %>%
        evaluate(test_x,
                 test_y)
    
    recall = c(recall, eval[5])
    accuracy = c(accuracy, eval[2])
    precision = c(precision, eval[4])
    auc = c(auc, eval[3])
    loss = c(loss, eval[1])
}
models = c(models, "Modelo 1")
layers = c(layers, 2)

gen_acc = c(gen_acc, mean(accuracy))
gen_auc = c(gen_auc, mean(auc))
gen_rec = c(gen_rec, mean(recall))
gen_pre = c(gen_pre, mean(precision))
gen_loss = c(gen_loss, mean(loss))
gen_lr = c(gen_lr, lr)
gen_ep = c(gen_ep, ep)