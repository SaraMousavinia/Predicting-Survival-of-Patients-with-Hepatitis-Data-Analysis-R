library(MASS)
library(dplyr)
library(glmnet)
library(randomForest)
library(gbm)
library(caret)
library(mice)
library(class)
library(pROC)
library(doParallel)
library(e1071)
library(obliqueRF)

# Loading dataset
hepatitis_data <- read.csv("F:\\Predictive modeling\\CourseProject\\Hepatitis_csv.csv", header = TRUE, sep = ",")
summary(hepatitis_data)
View(hepatitis_data)
names(hepatitis_data)

# Encoding Categorical Variables
hepatitis_data$sex <- factor(hepatitis_data$sex, 
                             levels = c("female","male"), 
                             labels =c(0,1))

hepatitis_data$steroid <- factor(hepatitis_data$steroid, 
                                 levels = c("True","False"), 
                                 labels = c(1,2))

hepatitis_data$antivirals <- factor(hepatitis_data$antivirals, 
                                    levels = c("True","False"), 
                                    labels = c(1,2))

hepatitis_data$fatigue <- factor(hepatitis_data$fatigue, 
                                 levels = c("True","False"), 
                                 labels = c(1,2))

hepatitis_data$malaise <- factor(hepatitis_data$malaise, 
                                 levels = c("True","False"), 
                                 labels = c(1,2))

hepatitis_data$anorexia <- factor(hepatitis_data$anorexia, 
                                  levels = c("True","False"), 
                                  labels = c(1,2))

hepatitis_data$liver_big <- factor(hepatitis_data$liver_big, 
                                   levels = c("True","False"), 
                                   labels = c(1,2))

hepatitis_data$liver_firm <- factor(hepatitis_data$liver_firm, 
                                    levels = c("True","False"), 
                                    labels = c(1,2))

hepatitis_data$spleen_palpable <- factor(hepatitis_data$spleen_palpable, 
                                         levels = c("True","False"), 
                                         labels = c(1,2))

hepatitis_data$spiders <- factor(hepatitis_data$spiders, 
                                 levels = c("True","False"), 
                                 labels = c(1,2))

hepatitis_data$ascites <- factor(hepatitis_data$ascites, 
                                 levels = c("True","False"), 
                                 labels = c(1,2))

hepatitis_data$varices <- factor(hepatitis_data$varices, 
                                 levels = c("True","False"), 
                                 labels = c(1,2))

hepatitis_data$histology <- factor(hepatitis_data$histology, 
                                   levels = c("True","False"), 
                                   labels = c(1,2))

hepatitis_data$class <- factor(hepatitis_data$class)

# Encoding numeric variables (some of them were disceret some others were continuous)
hepatitis_data$age <- as.numeric(hepatitis_data$age)
hepatitis_data$bilirubin <- as.numeric(hepatitis_data$bilirubin)
hepatitis_data$alk_phosphate <- as.numeric(hepatitis_data$alk_phosphate)
hepatitis_data$sgot <- as.numeric(hepatitis_data$sgot)
hepatitis_data$albumin <- as.numeric(hepatitis_data$albumin)
hepatitis_data$protime <- as.numeric(hepatitis_data$protime)

View(hepatitis_data)

# Handling missing values
hepatitis_data[hepatitis_data == ""] <- NA  # Replace blanks by NA


sum(is.na(hepatitis_data))                       #number of missing value all over the dataset
p <- function(x) {sum(is.na(x))/length(x)*100}   #the percentage of missing values at each column
apply(hepatitis_data, 2, p)
hepatitis_data[!complete.cases(hepatitis_data),]
hepatitis_data[complete.cases(hepatitis_data),]

# Clean the missing values
cleaned_hepatitis<- na.omit(hepatitis_data)
sum(is.na(cleaned_hepatitis))

# Check the importance of variables
ZeroVar <- nearZeroVar(cleaned_hepatitis,saveMetrics=TRUE)
nzv_sorted <- arrange(ZeroVar, desc(freqRatio))
nzv_sorted # non of variables have zero or near zero variances

# Drop columns 3,10
cleaned_hepatitis_reduced<- subset(cleaned_hepatitis, select = -c(3,4,9,10)) # dropping the 8 didnt change the accuracy

set.seed(2021) 
# Split dataset
inTraining <- createDataPartition(cleaned_hepatitis_reduced$class, ## indicate the outcome - helps in balancing the partitions
                                  p = .8, ## proportion used in training+ testing subset
                                  list = FALSE)
training <- cleaned_hepatitis_reduced[ inTraining,]
holdout  <- cleaned_hepatitis_reduced[-inTraining,]

# Scaling the training set (only the numeric variables, dummy variables dont need standardization)
preProcValues <- preProcess(training[c(1,10,11,12,13,14)], method = c("center", "scale"))
trainTransformed <- predict(preProcValues, training)

# Holdout set preparation
# Scaling the holdout set (only the numeric variables, dummy variables dont need standardization)
preProcValues <- preProcess(holdout[c(1,10,11,12,13,14)], method = c("center", "scale"))
holdoutTransformed <- predict(preProcValues, holdout)

############################################################################################
# 1. Logistic- glmboost
############################################################################################
set.seed(2021)
# Parameter tuning
fitControl <- trainControl(method = "repeatedcv", ## indicate that we want to do k-fold CV
                           number = 10, ## k = 10
                           repeats = 3, ## and repeat 10-fold CV 10 times
                           ## Estimate class probabilities
                           savePredictions = TRUE,
                           classProbs = TRUE)

set.seed(2021)
# Model Fitting
glmBoostModel <- train(class ~ ., data=trainTransformed, 
                       method = "glmboost", 
                       trControl = fitControl,
                       tuneLength=10)

# Accuracy on Trainingset
mean(glmBoostModel$results$Accuracy)

# Make a Prediction on Holdout Set
glmBoostModel.pred <- predict(glmBoostModel, # predict using the fitted model
                              holdoutTransformed,
                              type = "raw") ## produce only the predicted values

# Confusion Matrix
confusionMatrix(data = glmBoostModel.pred, reference = holdoutTransformed$class, positive="live")
postResample(pred = glmBoostModel.pred, obs = holdoutTransformed$class)

# Importance of Variables
varImp(glmBoostModel)


############################################################################################
# 2. LDA and QDA 
############################################################################################
set.seed(2021)

# Parameter Tuning
fitControl <- trainControl(method = "repeatedcv", ## indicate that we want to do k-fold CV
                           number = 10, ## k = 10
                           repeats = 3, ## and repeat 10-fold CV 10 times
                           ## Estimate class probabilities
                           classProbs = TRUE)
# Model Fitting
set.seed(2021)
ldamodel <- train(class ~ ., 
                  data=trainTransformed, 
                  method="lda",
                  trControl = fitControl)

# Accuracy of Training Set
mean(ldamodel$results$Accuracy)

# Make a Prediction on Holdout Set
ldamodel_pred <- predict(ldamodel, 
                         holdoutTransformed,
                         type = "raw") ## produce only the predicted values

# Confusion Matrix
confusionMatrix(data = ldamodel_pred, reference = holdoutTransformed$class, positive="live")
postResample(pred = ldamodel_pred, obs = holdoutTransformed$class)

# Importance of Variables
varImp(ldamodel)

# QDA
fitControl <- trainControl(method = "repeatedcv", ## indicate that we want to do k-fold CV
                           number = 10, ## k = 10
                           repeats = 10, ## and repeat 10-fold CV 10 times
                           classProbs = TRUE)
set.seed(2021)
qdamodel <- train(class ~ ., data=trainTransformed, method="qda",trControl = fitControl)
qdamodel

varImp(ldamodel)

## Could not run qda since the dataset is small for qda, I need to eliminate some variables

############################################################################################
# 3. Knn
############################################################################################
set.seed(2021)

# Parameter Tuning
knnGrid <-  expand.grid(k = c(1:3))
fitControl <- trainControl(method = "repeatedcv",
                           number = 10, 
                           repeats = 3, # uncomment for repeatedcv 
                           classProbs = TRUE)

# Model Fitting
knnmodel <- train(class ~ ., 
                  data = trainTransformed[,-1], 
                  method = "knn",  
                  trControl = fitControl,
                  tuneGrid = knnGrid)
plot(knnmodel)

# Accuracy of Training Set
mean(knnmodel$results$Accuracy)

# Make a Prediction on Holdout Set
knnmodel_pred <- predict(knnmodel, holdoutTransformed)

# Confusion Matrix
confusionMatrix(data = knnmodel_pred, reference = holdoutTransformed$class, positive="live",mode = "prec_recall")
postResample(pred = knnmodel_pred, obs = holdoutTransformed$class)

# Importance of the Variables
varImp(knnmodel)

############################################################################################
# 4. Random Forest
############################################################################################
set.seed (2021)

# Parameter Tuning
fitControl <- trainControl(method = "repeatedcv",
                           number = 10, 
                           repeats = 3,
                           classProbs = TRUE)

rfGrid <- expand.grid(mtry=c(1:3))

set.seed(2021)

# Model Fitting
randomforestmodel <- train(class ~ .,
                           data = trainTransformed, 
                           method = "rf", 
                           trControl = fitControl, 
                           n.trees=seq(500,1500,by=200),
                           verbose = FALSE, 
                           tuneGrid = rfGrid,
)

# Accuracy of Training Set
mean(randomforestmodel$results$Accuracy)

# Make a Prediction on Holdout Set
randomforestmodel_pred <- predict(randomforestmodel, holdoutTransformed)

# Confusion Matrix
confusionMatrix(data = randomforestmodel_pred, reference = holdoutTransformed$class, positive="live",mode = "prec_recall")
postResample(pred = randomforestmodel_pred, obs = holdoutTransformed$class)

# Importance of Variables
varImp(randomforestmodel)


set.seed (2021)
# Parameter Tuning
fitControl <- trainControl(method = "repeatedcv",
                           number = 10, 
                           repeats = 3,
                           classProbs = TRUE)

rfGrid <- expand.grid(mtry=c(1:4))

set.seed(2021)
# Model Fitting
randomforestSVM <- train(class ~ .,
                         data = trainTransformed, 
                         method = "ORFsvm", 
                         trControl = fitControl, 
                         n.trees=seq(200,1500,by=100),
                         verbose = FALSE, 
                         tuneGrid = rfGrid,
)

# Accuracy of Training Set
mean(randomforestSVM$results$Accuracy)

# Make a Prediction on Holdout Set
randomforestSVM_pred <- predict(randomforestSVM, holdoutTransformed, type="raw")

# Confusion Matrix
confusionMatrix(data = randomforestSVM_pred, reference = holdoutTransformed$class, positive="live",mode = "prec_recall")
postResample(pred = randomforestSVM_pred, obs = holdoutTransformed$class)

# Importance of the Variables
varImp(randomforestSVM)

############################################################################################
# 5. Boosted Tree
############################################################################################
set.seed(2021) 
fitControl <- trainControl(method = "repeatedcv",
                           number = 10, 
                           repeats = 3,
                           classProbs = TRUE)
# Parameter Tuning
grid <- expand.grid(interaction.depth = seq(1:3),
                    shrinkage = seq(from = 0.01, to = 0.2, by = 0.01),
                    n.trees = seq(from = 10, to = 50, by = 10),
                    n.minobsinnode = seq(from = 5, to = 20, by = 5)
)

# Model Fitting
boostedtreemodel <- train(class ~ .,
                          data = trainTransformed,
                          method = "gbm",
                          trControl = fitControl,
                          verbose = FALSE, ## setting this to TRUE or excluding this leads to a lot of output
                          tuneGrid = grid)

# Accuracy of Training Set
boostedtreemodel$results
results<- data.frame(boostedtreemodel$results$Accuracy,boostedtreemodel$results$Kappa)
cleaned_results <- na.omit(results)
mean(cleaned_results$boostedtreemodel.results.Accuracy)


# Make a Prediction on Holdout Set
boostedtreemodel_pred <- predict(boostedtreemodel, holdoutTransformed)

# Confusion Matrix
confusionMatrix(data = boostedtreemodel_pred, reference = holdoutTransformed$class, positive="live")

# Importance of Variables
varImp(boostedtreemodel)

############################################################################################
# 6 , 7. SVM
############################################################################################
set.seed(2021)

######################## Model #1: Linear SVM #########################################
set.seed(2021)
# Parameter Tuning
grid <- expand.grid(C = c(0.01, 0.1, 10, 100, 1000))
fitControl <- trainControl(method = "repeatedcv",
                           number = 10, 
                           repeats = 3,
                           classProbs = TRUE)
# Model Fitting
svmlinearfitmodel <- train(class ~ .,
                           data = trainTransformed,
                           method = "svmLinear",
                           trControl = fitControl,
                           verbose = FALSE, ## setting this to TRUE or excluding this leads to a lot of output
                           tuneGrid = grid)

# Accuracy of Training Set
mean(svmlinearfitmodel$results$Accuracy)

# Make a Prediction on Holdout Set
svmlinearfitmodel_pred <- predict(svmlinearfitmodel, holdoutTransformed)

# Confusion Matrix
confusionMatrix(data=svmlinearfitmodel_pred, reference=holdoutTransformed$class,positive="live")
postResample(pred = svmlinearfitmodel_pred, obs = holdoutTransformed$class)

# Importance of Variables
varImp(svmlinearfitmodel)

######################## Model #2: Radial SVM #########################################
set.seed(2021)
# Parameter Tuning
gridsvm <- expand.grid(C = c(0.01, 0.1, 10, 100,200,500, 1000),
                       sigma = c(0.1,0.5, 1, 2, 3, 4))

fitControl <- trainControl(method = "repeatedcv",
                           number = 10, 
                           repeats = 3,
                           classProbs = TRUE)
# Model Fitting
svmradialmodel <- train(class ~ .,
                        data = trainTransformed,
                        method = "svmRadial",
                        trControl = fitControl,
                        verbose = FALSE, ## setting this to TRUE or excluding this leads to a lot of output
                        tuneGrid = gridsvm)

# Accuracy of Training Set
mean(svmradialmodel$results$Accuracy)


# Make a Prediction on Holdoutset
svmradialmodel_pred <- predict(svmradialmodel, holdoutTransformed)

# Confusion Matrix
confusionMatrix(data=svmradialmodel_pred, reference=holdoutTransformed$class,positive="live",mode = "prec_recall")
postResample(pred = svmradialmodel_pred, obs = holdoutTransformed$class)

# Importance of Variables
varImp(svmradialmodel)

############################################################################################
# Plots
############################################################################################
# 1. Logistic
trellis.par.set(caretTheme())
plot(glmBoostModel)

# 2. LDA
trellis.par.set(caretTheme())
plot(ldamodel)

# 3. KNN
trellis.par.set(caretTheme())
plot(knnmodel)

# 4. Random Forest 
trellis.par.set(caretTheme())
plot(randomforestmodel)

# 5. Boosted Tree
trellis.par.set(caretTheme())
plot(boostedtreemodel)

# 6. SVM-Linear
trellis.par.set(caretTheme())
plot(svmlinearfitmodel)

# 7. SVM-Radial
trellis.par.set(caretTheme())
plot(svmradialmodel)

############################################################################################
# Model Comparison
############################################################################################
set.seed(2021)
fitControl <- trainControl(method = "repeatedcv",repeats=3,number = 10,classProbs = TRUE)
ldamodel <- train(class ~ ., 
                  data=trainTransformed, 
                  method="lda",
                  trControl = fitControl,
                  metric="Kappa")
ldamodel$resample

set.seed (2021)
grid <- expand.grid(C = c(0.01, 0.1, 10, 100, 1000))
fitControl <- trainControl(method = "repeatedcv",repeats=3,number = 10,classProbs = TRUE)
svmlinearfitmodel <- train(class ~ .,
                           data = trainTransformed,
                           method = "svmLinear",
                           trControl = fitControl,
                           verbose = FALSE, ## setting this to TRUE or excluding this leads to a lot of output
                           tuneGrid = grid,
                           metric="Kappa")

svmlinearfitmodel$resample 

set.seed(2021)
fitControl <- trainControl(method = "repeatedcv",repeats=3,number = 10,classProbs = TRUE)
gridsvm <- expand.grid(C = c(0.01, 0.1, 10, 100,200,500, 1000,2000),
                       sigma = c(0.1,0.5, 1, 2, 3, 4))
svmradialmodel <- train(class ~ .,
                        data = trainTransformed,
                        method = "svmRadial",
                        trControl = fitControl,
                        verbose = FALSE, ## setting this to TRUE or excluding this leads to a lot of output
                        tuneGrid = gridsvm,
                        metric="Kappa")
svmradialmodel$resample

resampls <- resamples(list(LDA = ldamodel,
                          SVMLinear=svmlinearfitmodel,
                          SVMRadial =svmradialmodel),replace = FALSE)
resampls

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resampls, layout = c(3, 1))

names(theme1)

trellis.par.set(theme1)
bwplot(difValues, layout = c(3, 1))

# Create a matrix
conf.df <- matrix(c(0.8,0.8571, 0.9231,0.8889,
                    0.8667,0.8667,1.00,0.9286,
                    0.9333,0.9286,1.00,0.9630), ncol=4, byrow=TRUE)
colnames(conf.df) <- c('Accuracy','Precision','Recall', 'FScore')
rownames(conf.df) <- c('KNN','RandomForest','SVM-Radial')
conf.df <- as.table(conf.df)
conf.df

# Grouped Bar Plot
hepatitisbarplot <- barplot(conf.df,col = c("#00CC99", "#FFCC99","#FF9999"), 
                            beside = TRUE,
                            legend.text = c("KNN", "RF", "SVM"),
                            args.legend=list(cex=0.7,x="right"),
                            main="Accuracy over All Models",
                            xlab="Metrics",
                            ylab="Percentage", ylim=c(0,1.1))
percentages <- c(0.8,0.8667, 0.9333,
                 0.8571,0.8667,0.9286,
                 0.9231,1.0,1.0,
                 0.8889,0.9286,0.9630)
text(x = hepatitisbarplot, y = percentages, 
     label = paste(percentages*100,"%"), pos = 3, cex = 0.55, col = "black",srt=45)


