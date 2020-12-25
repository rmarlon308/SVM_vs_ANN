library(readr)
library(dplyr)
library(keras)
library(tensorflow)
library(parallel)


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


learning_rate = c(0.001, 0.005, 0.01)
epochs = c(50, 100, 150)

gen_acc = c()
gen_auc = c()
gen_pre = c()
gen_rec = c()
gen_loss = c()
gen_lr = c()
gen_ep = c()
models = c()
layers = c()

rand_data = all_data[sample(nrow(all_data)),]

folds = cut(seq(1, nrow(rand_data)), breaks = 10, labels = F)

for(lr in learning_rate){
    for(ep in epochs){
        mclapply(source("/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_2/ANN_models_2/model1.R"))
        mclapply(source("/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_2/ANN_models_2/model2.R"))
        mclapply(source("/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_2/ANN_models_2/model3.R"))
        mclapply(source("/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_2/ANN_models_2/model4.R"))
    }
}

scores_ann_2 = data.frame(models,layers, gen_auc, gen_acc, gen_ep, gen_loss, gen_lr, gen_pre, gen_rec)

write.csv(scores_ann_2, "/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_2/ANN_models_2/scores_ann_2.csv")

best_model = scores_ann_2 %>%
    arrange(desc(gen_auc))

best_topology = best_model[1,]
