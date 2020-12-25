library(keras)
library(readr)
library(dplyr)
library(tensorflow)
library(precrec)
library(parallel)

dataset <- read_csv("/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_1/dataset/wdbc.data", 
                    col_names = FALSE)
dataset = dataset %>%
    select(-1) %>%
    rename("diagnosis" = X2)

for(i in 2:ncol(dataset)){
    dataset[i] =  (dataset[i] - min(dataset[i]))/ (max(dataset[i]) - min(dataset[i])) * (1-0) + 0
}

x_data = dataset[, 2:ncol(dataset)]
y_data = dataset[, 1]

y_data$diagnosis = as.factor(y_data$diagnosis)

y_data$diagnosis = as.numeric(y_data$diagnosis) - 1 # M:1, B:0

#PCA
mean_x = colMeans(x_data)

for(i in 1:nrow(x_data)){
    x_data[i, ] = x_data[i, ] - mean_x
}

svd = svd(x_data)

eigenValues = svd$d^2/ sum(svd$d^2)

eigenVectors = svd$v

variance = cumsum(eigenValues)/sum(eigenValues)

k = NULL

for(i in 1:length(variance)){
    if(variance[i] >= 0.95){
        k = i
        print(k)
        break
    }
}

k_selected = eigenVectors[,1:k]

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
        mclapply(source("/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_1/ANN_models/model1.R"))
        mclapply(source("/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_1/ANN_models/model2.R"))
        mclapply(source("/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_1/ANN_models/model3.R"))
        mclapply(source("/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_1/ANN_models/model4.R"))
    }
}

scores = data.frame(models,layers, gen_auc, gen_acc, gen_ep, gen_loss, gen_lr, gen_pre, gen_rec)
        
write.csv(scores, "/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_1/ANN_models/scores.csv")

best_model = scores %>%
    arrange(desc(gen_auc))

best = best_model[1, ]