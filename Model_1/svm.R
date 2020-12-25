library(readr)
library(dplyr)
library(e1071)
library(precrec)
library(vtreat)
library(pROC)


dataset <- read_csv("/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_1/dataset/wdbc.data", 
                 col_names = FALSE)



dataset = dataset %>%
    select(-X1) %>%
    rename("diagnosis" = X2)

dataset$diagnosis = as.factor(dataset$diagnosis)

dataset %>%
    select(diagnosis) %>%
    group_by(diagnosis) %>%
    summarise(n = n())

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

final_data = cbind(y_data, proy_data)

#tunning
gamma_list = 5*10^(-2:2)
cost_list = c(0.01, 0.1, 1, 10, 100)
kernel_list = c("radial", "linear", "polynomial")

splitPlan = kWayCrossValidation(nRows = nrow(final_data), 10, NULL, NULL)

gen_acc = c()
gen_auc = c()
gen_pre = c()
gen_rec = c()

gen_gamma = c()
gen_cost = c()
gen_kernel = c()

for(gm in gamma_list){
    for(cos in cost_list){
        for(ker in kernel_list){
            recall = c()
            accuracy = c()
            precision = c()
            auc = c()
            
            for(i in 1:10){
                split = splitPlan[[i]]
                
                svm_model = svm(diagnosis~ ., data = final_data[split$train, ], type = "C-classification", 
                                kernel = ker, 
                                cost = cos, 
                                gamma = gm)
                
                pred = predict(svm_model, final_data[split$app, ])
                real = final_data[split$app, ]$diagnosis
                
                confusion = as.matrix(table(real, pred))
                
                
                diag = diag(confusion)
                colsums = apply(confusion, 2, sum)
                rowsums = apply(confusion, 1, sum)
                n = sum(confusion)
                
                recall[i] = diag / rowsums 
                precision[i] = diag / colsums
                accuracy[i] = sum(diag) / n 
                x = roc(as.numeric(real), (as.numeric(pred) - 1))
                auc[i] = x$auc
                
            }
            
            gen_acc = c(gen_acc, mean(accuracy))
            gen_auc = c(gen_auc, mean(auc))
            gen_pre = c(gen_pre, mean(precision))
            gen_rec = c(gen_rec, mean(recall))
            
            gen_gamma = c(gen_gamma, gm)
            gen_cost = c(gen_cost, cos)
            gen_kernel = c(gen_kernel, ker)
            
            
            
            
        }
    }
}

score_svm = data.frame(gen_kernel, gen_gamma, gen_cost, gen_acc, gen_auc, gen_pre, gen_rec)

write_csv(score_svm, "/home/marlon/mainfolder/marlon/USFQ/DataMining/10_FinalProject/proyectoFinal/Model_1/score_svm.csv")

best_model = score_svm %>%
    arrange(desc(gen_auc)) 

best_model = best_model[1, ]

best_gamma = best_model[, 2]
best_cost = best_model[,3]
best_kernel = best_model[,1]

splitPlan = kWayCrossValidation(nRows = nrow(final_data), 10, NULL, NULL)

recall = c()
accuracy = c()
precision = c()
auc = c()
pred_vector = rep(0, nrow(final_data))

for(i in 1:10){
    split = splitPlan[[i]]
    
    svm_model = svm(diagnosis~ ., data = final_data[split$train, ], type = "C-classification", kernel = best_kernel, 
               cost = best_cost, 
               gamma = best_gamma)
    
    pred_vector[split$app] = predict(svm_model, final_data[split$app, ])
    
    pred = predict(svm_model, final_data[split$app, ])
    
    real = final_data[split$app, ]$diagnosis
    
    confusion = as.matrix(table(real, pred))
    
    
    diag = diag(confusion)
    colsums = apply(confusion, 2, sum)
    rowsums = apply(confusion, 1, sum)
    n = sum(confusion)
    
    recall[i] = diag / rowsums 
    precision[i] = diag / colsums
    accuracy[i] = sum(diag) / n
    x = roc(as.numeric(real), (as.numeric(pred) - 1))
    auc[i] = x$auc
}


sprintf("Accuracy   Mean: %f   SD: %f", mean(accuracy), sd(accuracy))
sprintf("Precision   Mean: %f   SD: %f", mean(precision), sd(precision))
sprintf("Recall   Mean: %f   SD: %f", mean(recall), sd(recall))
sprintf("AUC   Mean: %f   SD: %f", mean(auc), sd(auc))

real_vector = final_data$diagnosis

table(pred_vector, real_vector)


precrec_obj <- evalmod(scores = pred_vector, labels = real_vector)
plot(precrec_obj)
precrec_obj

#Linear Split
svm_model = svm(diagnosis~ ., data = final_data[split$train, ], type = "C-classification", kernel = best_kernel, 
                cost = best_cost, 
                gamma = best_gamma)

plot(svm_model, final_data[split$train, ], V1 ~ V2)



