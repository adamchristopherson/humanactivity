library(caret)
setwd("~/Desktop/R_class/machine_learning/")
## read in data and ensure NAs are categorized properly
training <- read.csv("pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
testing <- read.csv("pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))

## remove the variables that are mostly NAs
NAs <- sapply(training, function(x) mean(is.na(x))) > .9
training <- training[, NAs==FALSE]

## remove variables that are not predictive
trainingTrain <- training[,-(1:6)]
str(trainingTrain)

## split training set into mytraining and mytesting (for cross validation)
set.seed(12345)
inTrain <- createDataPartition(y=trainingTrain$classe, p=0.75, list=FALSE)
mytraining <- trainingTrain[inTrain,]
mytesting <- trainingTrain[-inTrain,]

## fit a tree
modFitTree <- train(classe ~ ., method = "rpart", data=mytraining)

## plot the dendogram
library(rattle)
fancyRpartPlot(modFitTree$finalModel)

## predict on cross val set
predTree <- predict(modFitTree, newdata=mytesting)
confusionMatrix(predTree, mytesting$classe)

## let's try a random forest - this takes too long
# modFitForest <- train(classe~., method="rf", data=mytraining, prox=TRUE)

## take a smaller subset, and train a tree to obtain important variables
inTrain2 <- createDataPartition(y=mytraining$classe, p=0.1, list=FALSE)
mytrainsmall <- mytraining[inTrain2,]

variableImportanceTest <- train(classe~., method="rf", data=mytrainsmall, prox=TRUE)

VI <- data.frame(varImp(variableImportanceTest)$importance)
VI$vars <- row.names(VI)
row.names(VI) <- NULL

preds <- head(VI[order(VI$Overall, decreasing=TRUE),], n=20)
predNames <- preds$vars
predNames2 <- paste(predNames, collapse=" + ")
predNamesForm <- as.formula(paste"classe ~ ", predNames2)

## now train the larger training set using only these top 20 variables
mytrainlarger <- mytraining[-inTrain2,]
modFitForest <- train(predNamesForm, method="rf", data=mytrainlarger, prox=TRUE)

predForest <- predict(modFitForest, newdata=mytesting)
confusionMatrix(predForest, mytesting$classe)

predict(modFitForest, newdata=testing)
