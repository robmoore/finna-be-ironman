#http://groupware.les.inf.puc-rio.br/har
#https://archive.ics.uci.edu/ml/datasets/Wearable+Computing%3A+Classification+of+Body+Postures+and+Movements+(PUC-Rio)

library(caret)

# enable multi-core processing
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

preProcessData = function(x){
  data <- read.csv(x, na.strings = c("NA", "", "#DIV/0!"))
  
  # Drop colums that are mostly NAs (> 97.5%)
  data <- data[,colSums(is.na(data)) < (nrow(data) * .975)]
  # Drop bookkeeping data
  data <- data[,-(1:7)] 
  
  # Use the columns with the most variation
  # WAS: data <- data[,grep(paste(c("roll_", "yaw_", "pitch_", "classe"), collapse="|"), names(data), perl=TRUE)]
  data[,grep(paste(c("^roll_", "^yaw_", "^pitch_", "classe"), collapse="|"), names(data), perl=TRUE)]
}

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n) {
    filename = paste0("results/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

set.seed(13413)

pml_training.data <- preProcessData("data/pml-training.csv")

#60% training
trainDP <- createDataPartition(pml_training.data$classe, p=0.6, list=FALSE)
train.set <- pml_training.data[trainDP,]
other.set <- pml_training.data[-trainDP,]
#20% test
#20% validation
otherDP  <- createDataPartition(other.set$classe, p=0.5, list=FALSE)
test.set <- other.set[otherDP,]
validation.set <- other.set[-otherDP,]

# Create model
modFit <- train(classe ~ ., data=train.set, method="rf", trControl = trainControl(method = "cv", number = 5))
# Should we use the valiation set instead?
pred.train <- predict(modFit, test.set[,-length(test.set)])

test.rate <- length(which(pred.train == test.set$classe)) / length(pred.train)
#[1] 0.9933724

pred.validation <- predict(modFit, validation.set[,-length(validation.set)])

validation.rate <- length(which(pred.validation == validation.set$classe)) / length(pred.validation)

pml_testing.data <- preProcessData("data/pml-testing.csv")
pml_testing.pred <- predict(modFit, pml_testing.data)

dir.create("results")
pml_write_files(rep("A", 20))
pml_write_files(pml_testing.pred)

---

#inTraining <- createDataPartition(mydata$FLAG, p=0.6, list=FALSE)
#training.set <- mydata[inTraining,]
#Totalvalidation.set <- mydata[-inTraining,]
## This will create another partition of the 40% of the data, so 20%-testing and 20%-validation
#inValidation <- createDataPartition(Totalvalidation.set$FLAG, p=0.5, list=FALSE)
#testing.set <- Totalvalidation.set[inValidation,]
#validation.set <- Totalvalidation.set[-inValidation,]

nzv <- nearZeroVar(data, saveMetrics=TRUE)
nzv[order(-nzv$percentUnique),] 

# Random forest usage for this case:
# http://bigcomputing.blogspot.com/2014/10/an-example-of-using-random-forest-in.html
