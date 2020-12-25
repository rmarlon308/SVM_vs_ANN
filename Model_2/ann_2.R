library(readr)
library(dplyr)
library(keras)
library(tensorflow)


testing = read_csv("/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_2/dataset/testing.csv")
training = read_csv("/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_2/dataset/training.csv")

dataset = rbind(training, testing)

dataset$class = as.factor(dataset$class)

dataset$class = as.numeric(dataset$class) - 1

#Normalize the data
summary(dataset)

for(i in 2:ncol(dataset)){
    dataset[i] =  (dataset[i] - min(dataset[i]))/ (max(dataset[i]) - min(dataset[i])) * (1-0) + 0
}

summary(dataset)

x_data = dataset[, 2:ncol(dataset)]

y_data = dataset[, 1]

#Dimensionality reduction
mean_row = colMeans(x_data)

for(i in 1:nrow(x_data)){
    x_data[i, ] = x_data[i, ] - mean_row
}

svd = svd(x_data)

eigVectors = svd$v

eigValues = svd$d^2/sum(svd$d^2)

variance = cumsum(eigValues)/sum(eigValues)

k = NULL

for(i in 1:length(variance)){
    if(variance[i] >= 0.95){
        k = i
        print(k)
        break
    }
}

k_selected = eigVectors[, 1:k]

proy_data = as.data.frame(as.matrix(x_data) %*% as.matrix(k_selected))

all_data = cbind(y_data, proy_data)


rand_data = all_data[sample(nrow(all_data)),]

#Del csv sacar el mejor modelo

folds = cut(seq(1, nrow(rand_data)), breaks = 10, labels = F)

recall = c()
accuracy = c()
precision = c()
auc = c()
loss = c()
pred_vector = rep(0, nrow(all_data))
real_vector = rand_data$class
history = c()

for(i in 1:10){
    test_index = which(folds == i, arr.ind = T)
    
    test_x = as.matrix(rand_data[test_index, 2:ncol(rand_data)])
    test_y = rand_data[test_index, 1]
    
    train_x = as.matrix(rand_data[-test_index, 2:ncol(rand_data)])
    train_y = rand_data[-test_index, 1]
    
    test_y = to_categorical(test_y)
    train_y = to_categorical(train_y)
    
    model = keras_model_sequential()
    
    model %>%
        layer_dense(units = 15, activation = 'relu', input_shape = c(ncol(train_x))) %>%
        layer_dense(units = 4, activation = 'softmax')
    
    model %>%
        compile(loss = 'categorical_crossentropy',
                optimizer = optimizer_adam(lr = 0.001),
                metrics = c('accuracy', tf$keras$metrics$AUC(), tf$keras$metrics$Precision(), tf$keras$metrics$Recall()))
    
    mymodel = model %>%
        fit(train_x,
            train_y,
            epochs = 150,
            batch_size = 32,
            validation_split = 0.2
        )
    
    eval = model %>%
        evaluate(test_x,
                 test_y)
    
    history = c(history, mymodel)
    recall = c(recall, eval[5])
    accuracy = c(accuracy, eval[2])
    precision = c(precision, eval[4])
    auc = c(auc, eval[3])
    loss = c(loss, eval[1])
    
    pred = model %>% predict_classes(test_x)
    
    pred_vector[test_index] = pred
    
}

sprintf("Accuracy   Mean: %f   SD: %f", mean(accuracy), sd(accuracy))
sprintf("Precision   Mean: %f   SD: %f", mean(precision), sd(precision))
sprintf("Recall   Mean: %f   SD: %f", mean(recall), sd(recall))
sprintf("AUC   Mean: %f   SD: %f", mean(auc), sd(auc))
sprintf("Loss   Mean: %f   SD: %f", mean(loss), sd(loss))

table(pred_vector, real_vector)

#Grafica Loss

data_loss = NULL
data_loss_val = NULL

for(i in seq(2, 20, by = 2)){
    data_loss = rbind(data_loss, as.vector(history[i]$metrics$loss))
    data_loss_val = rbind(data_loss_val, as.vector(history[i]$metrics$val_loss))
}

mean_loss = colMeans(data_loss)
mean_loss_val = colMeans(data_loss_val)

loss_data = data.frame(mean_loss, mean_loss_val)

library(ggplot2)

ggplot(loss_data) +
    geom_line(aes(x = 1:nrow(loss_data), y = mean_loss), color = "blue") +
    geom_line(aes(x =  1:nrow(loss_data), y = mean_loss_val), color = "green")+
    xlab("Epochs") + ylab("Loss") + labs(title = "Loss")

library(pROC)
library(caret)

roc.multi = multiclass.roc(real_vector, pred_vector, levels = c(0,1,2,3))

auc(roc.multi)
rs <- roc.multi[['rocs']]
plot.roc(rs[[1]])
sapply(2:length(rs),function(i) lines.roc(rs[[i]],col=i))

