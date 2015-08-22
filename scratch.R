# Web site for study data is based on: http://groupware.les.inf.puc-rio.br/har
# Original data set (NOT the same as what we were provided): https://goo.gl/keVQRv

library(caret)

# enable multi-core processing
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

preProcessData = function(x) {
  data <- read.csv(x, na.strings = c("NA", "", "#DIV/0!"))
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

# Use the columns with the most variation
subsetData = function(data) {
  cat("[subsetData] Original dimensions: ", 
      dim(data), 
      "\n")
  
  data <- data[, grep(paste(c("^(roll|yaw|pitch)_",
                              "classe"), 
                            collapse = "|"), 
                      names(data),
                      perl = TRUE)]
    
  ## Find columns highly correlated with last variable (ie, yaw_forearm)
  #cor.matrix <- cor(data[,-length(data)])
  #highlyCorrelated <- findCorrelation(cor.matrix,  cutoff = 0.75)
  ## Drop highly correlated columns
  #data <- data[, -highlyCorrelated] # drops yaw_belt from this set at .75 cutoff
  
  cat("[subsetData] Resulting dimensions: ", 
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

verbosePredict = function(modFit, data) {
  pred <- predict(modFit, data[,-length(data)])
  
  # OOS rate
  #cat("Accuracy on data set: ", 
  #    length(which(pred == data$classe)) / length(data), 
  #    "\n")
  
  confusionMatrix(data$classe, pred)
}

pml_training.data <- preProcessData("data/pml-training.csv")

nzv <- nearZeroVar(pml_training.data, saveMetrics = TRUE)
nzv[order(-nzv$percentUnique),] 

pml_training.data <- subsetData(pml_training.data)

set.seed(13413)

# 60% training
trainDP <- createDataPartition(pml_training.data$classe, p = 0.6, list = FALSE)
train.set <- pml_training.data[trainDP,]
other.set <- pml_training.data[-trainDP,]
# 20% test and 20% validation
otherDP  <- createDataPartition(other.set$classe, p = 0.5, list = FALSE)
test.set <- other.set[otherDP,]
validation.set <- other.set[-otherDP,]

# Create model and test it against the validation set
startTime <- Sys.time()
modFit <- train(classe ~ ., 
                data=train.set, 
                method="rf", 
                trControl = trainControl(method = "cv", 
                                         number = 5))
Sys.time() - startTime # Time required to train model

# Test it against the validation set
verbosePredict(modFit, validation.set)

# Test it against the test set
verbosePredict(modFit, test.set)

# Generate predictions on provided test data
pml_testing.data <- subsetData(preProcessData("data/pml-testing.csv"))
pml_testing.pred <- predict(modFit, pml_testing.data)

# Write out results of testing predication
pml_write_files(pml_testing.pred)

---

nzv <- nearZeroVar(data, saveMetrics=TRUE)
nzv[order(-nzv$percentUnique),] 

look at varImp and plot (varImpPlot?)
can we plot nzv?

https://github.com/vqv/ggbiplot

http://stats.stackexchange.com/questions/30691/how-to-interpret-oob-and-confusion-matrix-for-random-forest
http://www.r-bloggers.com/part-3-random-forests-and-model-selection-considerations/
  
