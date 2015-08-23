# Web site for study data is based on: http://groupware.les.inf.puc-rio.br/har
# Original data set (NOT the same as what we were provided): https://goo.gl/keVQRv

library(caret)

# enable multi-core processing
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

readData = function(fn) {
  read.csv(paste0("data/", fn), na.strings = c("NA", "", "#DIV/0!"))
}

preProcessData = function(data) {
  cat("[preProcessData] Original dimensions: ", 
      dim(data), 
      "\n")
  
  # Drop colums that are mostly NAs (> 97.5%)
  data <- data[, colSums(is.na(data)) < (nrow(data) * .975)]
  # Drop book-keeping columns
  data <- data[, -(1:7)] 
  
  cat("[preProcessData] Resulting dimensions: ", 
      dim(data), 
      "\n")

  data
}

# Function based on that provided by instructors to write out results
pml_write_files = function(x) {
  dir.create("results", showWarnings = FALSE)
  n = length(x)
  for(i in 1:n) {
    filename = paste0("results/problem_id_", i, ".txt")
    write.table(x[i], file=filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
}

timedTrain = function(data) {
  # Create model and test it against the validation set
  startTime <- Sys.time()
  modFit <- train(classe ~ ., 
                  data=data, 
                  method="rf", 
                  trControl = trainControl(method = "cv", 
                                           number = 5))
  print(Sys.time() - startTime) # Time required to train model
  
  modFit
}

#summary(readData("pml-training.csv"))

pml_training.data <- preProcessData(readData("pml-training.csv"))

set.seed(13413)

# 60% training
trainDP <- createDataPartition(pml_training.data$classe, p = 0.6, list = FALSE)
train.set <- pml_training.data[trainDP,]
other.set <- pml_training.data[-trainDP,]
# 20% test and 20% validation
otherDP  <- createDataPartition(other.set$classe, p = 0.5, list = FALSE)
test.set <- other.set[otherDP,]
validation.set <- other.set[-otherDP,]

nzv <- nearZeroVar(train.set, saveMetrics = TRUE)
nzv[order(-nzv$percentUnique),] 
# Top 10 columns based on unique values > 10%
nzvColumns <- row.names(nzv)[order(-nzv$percentUnique)][1:16]
train.set.nzvColumns <- train.set[, c(nzvColumns, "classe")]

# Find columns highly correlated with last variable (ie, yaw_forearm)
cor.matrix <- cor(train.set[, -length(train.set)]) # eval all but classe column
highlyCorrelated <- findCorrelation(cor.matrix)
names(train.set)[highlyCorrelated] # 0 == Nothing found at .9 cutoff

# Create model on all columns for use with varImp
modFit <- timedTrain(train.set)
varImpPlot(modFit$finalModel)
vi <- varImp(modFit$finalModel)
# Top 7 important columns as identified by varImp
viColumns <- row.names(vi)[order(-vi$Overall)][1:7]
train.set.viColumns <- train.set[, c(viColumns, "classe")]

# Create model and test it against the validation set
modFit.nzvColumns <- timedTrain(train.set.nzvColumns)
predict.nzvColumns <- predict(modFit.nzvColumns, validation.set[, nzvColumns])
confusionMatrix(validation.set$classe, predict.nzvColumns)

modFit.viColumns <- timedTrain(train.set.viColumns)
predict.viColumns <- predict(modFit.viColumns, validation.set[, viColumns])
confusionMatrix(validation.set$classe, predict.viColumns)

# Test using viColumns
predict.viColumns <- predict(modFit.viColumns, test.set[, viColumns])
confusionMatrix(test.set$classe, predict.viColumns)

# Generate predictions on provided test data
pml_testing.data <- preProcessData(readData("pml-testing.csv"))
# TODO: only predict against vi columns
pml_testing.pred <- predict(modFit.viColumns, pml_testing.data)

# Write out results of testing predication
pml_write_files(pml_testing.pred)