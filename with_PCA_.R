
install.packages('tidyverse') #  read_csv function 
install.packages("skimr") #  Skim function
install.packages('caret') # Data Partitioning
library(tidyverse)
library(skimr) 
library(caret)

# Loading the dataset

Breast_Cancer.df <- read_csv("~/Downloads/Breast Cancer.csv")

# Investigate the data:
dim(Breast_Cancer.df)  # find the dimension of data frame
head(Breast_Cancer.df)  # shows the first six rows
View(Breast_Cancer.df)  # shows all the data in a new tab
skim(Breast_Cancer.df)  # shows the complete data summary along with hist plots


# Data Pre-processing 

# Step 1: Checked for Null Values

is.na(Breast_Cancer.df) # No Null Values were found

# Step 2: Converting the malign to 1 and benign to 0 for the diagnosis column

Breast_Cancer.df$diagnosis <- ifelse(Breast_Cancer.df$diagnosis == 'M', 1, 0)

# Step 3: Removing the ID column which is not required for the analysis

Breast_Cancer.df <- Breast_Cancer.df[, -1]

# Step 4: Correlation

corr_mat <- cor(Breast_Cancer.df[,3:ncol(Breast_Cancer)])
corrplot(corr_mat, method = 'color', order = "hclust", tl.cex = 0.5, tl.col = 'darkblue', addrect = 8, col = COL2('PuOr', 10))



# Converted the Numeric values to Factors for ease of calculations in later stage
#Breast_Cancer.df$diagnosis <- factor(Breast_Cancer.df$diagnosis, levels = c(0,1))
#class(Breast_Cancer.df$diagnosis) # Factor


table(Breast_Cancer.df$diagnosis)

# Step : Splitting the data in Training and Testing 

train.index <- createDataPartition(Breast_Cancer.df$diagnosis, p=0.7, list=F)
trainData <- Breast_Cancer.df[train.index,]
testData <- Breast_Cancer.df[-train.index,]

table(trainData$diagnosis) # 0- 252, 1-147


# Step : Normalizing the data 

norm.values <- preProcess(trainData, method= 'range')
train.norm.df <- predict(norm.values, trainData)
test.norm.df <- predict(norm.values, testData)

# Up Sampling the data

exclude_indices <- which(colnames(train.norm.df) != "diagnosis")
up_train <- upSample(
  x = train.norm.df[, exclude_indices, drop = FALSE],
  y = as.factor(train.norm.df$diagnosis)
)

table(train.norm.df$diagnosis)
names(up_train)[names(up_train) == "Class"] <- "diagnosis"
table(up_train$diagnosis) # 0s -252, 1s - 252


# Step 5: Creating a Principal Component Analysis

pca_bc.df <- prcomp(up_train[,-31], center = TRUE, scale = TRUE)
head(pca_bc.df$x)
summary(pca_bc.df) # Till PC17 ~99% variance is captured 

# read about this
screeplot(pca_bc.df, type="lines")

#did with uptrain

diag <- up_train$diagnosis
length(diag)
tarvar <- up_train[, colnames(up_train) != "diagnosis"]
length(tarvar)

# Extracting PCA components

pca_comp <- predict(pca_bc.df, tarvar)


# Selecting Number of Comp
#for 14 components, because of the 0.98
#CHALLENGE
numc <- sum(cumsum(pca_bc.df$sdev^2) / sum(pca_bc.df$sdev^2) < 0.98) + 1

select_Comp <- pca_comp[, 1:numc]

# Combine selected components with the target variable

pca_data <- data.frame(select_Comp, diagnosis = diag)

table(pca_data$diagnosis)

View(pca_data)

# Model 1: Logistics Regression 
logist.reg <- glm(diagnosis ~ ., family = 'binomial', data = pca_data)
AIC(logist.reg)

#pred_logist <- predict(logist.reg, newdata = test.norm.df, type = 'response')

# Creating a test_pca component
test_pca_components <- predict(pca_bc.df, test.norm.df
                               [, colnames(test.norm.df) != "diagnosis"])[, 1:numc]

test_pca_data <- data.frame(test_pca_components, diagnosis = test.norm.df$diagnosis)

# Make predictions
pred_pca <- predict(logist.reg, newdata = test_pca_data, type = 'response')

# Evaluate the model
y_pred_pca <- ifelse(pred_pca > 0.5, 1, 0)
class(y_pred_pca)
y_act <- test_pca_data$diagnosis

mean(y_pred_pca == y_act)
class(y_act)
y_act <- as.factor(y_act)
y_pred_pca <- as.factor(y_pred_pca)


# Confusion Matrix
confusionMatrix(as.factor(y_pred_pca), as.factor(y_act), positive = "1")

--------------------------------------------------
  #  Did Logistic Regression with up Sampling on PCA 
  --------------------------------------------------
  
  # KNN
  
  install.packages("caret") #for confusion matrix
library("caret")
install.packages("FNN")
library(FNN)

View(pca_data)
View(test_pca_data)

# Model 2: KNN

nn <- knn(train = pca_data[, - 15], test =test_pca_data[,-15], 
          cl =  pca_data$diagnosis, k = 2)

pca_data$diagnosis <- as.factor(pca_data$diagnosis)
test_pca_data$diagnosis <- as.factor(test_pca_data$diagnosis)
conf_matrix <- confusionMatrix(nn,test_pca_data$diagnosis, positive = "1")


accuracy.df <- data.frame(k = seq(1, 12, 1), accuracy = rep(0, 12))
for (i in 1:12) {
  nn <- knn(train = pca_data[, -15], test = test_pca_data[, -15], 
            cl = pca_data$diagnosis, k = i)
  
  # Calculate confusion matrix and extract accuracy
  cm <- confusionMatrix(nn, test_pca_data$diagnosis, positive = "1")
  accuracy.df[i, 2] <- cm$overall["Accuracy"]
}

# Print the resulting data frame
print(accuracy.df)
-------------------------------------------
  
  # Train kNN model with k=4
  knn_modelk4 <- knn(train = pca_data[, -15], test = test_pca_data[, -15], 
                     cl = pca_data$diagnosis, k = 4)
# Create confusion matrix
conf_matrix <- confusionMatrix(knn_modelk4, test_pca_data$diagnosis)





-----------------------------------------------
  
  # Random Forests 
  install.packages("randomForest")
library(randomForest)
install.packages("partykit")
library(partykit)


model_summary <- data.frame(Model = character(0), Accuracy = numeric(0))

# Random Forests

train.rf <- randomForest(as.factor(pca_data$diagnosis) ~ .,
                         data = pca_data,
                         ntree = 5,
                         importance = TRUE)
accuracy_rf <- confusionMatrix(predict(train.rf, newdata = test_pca_data), test_pca_data$diagnosis, positive = "1")$overall["Accuracy"]
model_summary <- rbind(model_summary, c("Random Forests", accuracy_rf))
predrft<- as.factor(predict(train.rf, newdata = test_pca_data))

confusionMatrix(as.factor(predrft), factor(test_pca_data$diagnosis), positive='1')

## variable importance plot
varImpPlot(train.rf, type = 1)

------------------------------------------
  install.packages("e1071")
library(e1071) 

# Naive Bayes model
naive_bayes_model <- naiveBayes(diagnosis ~ ., data = pca_data)

# Make predictions on the test data
predictions_nb <- predict(naive_bayes_model, newdata = test_pca_data)

# Evaluate the Naive Bayes model
confusionMatrix(predictions_nb, test_pca_data$diagnosis, positive = "1") 

install.packages("neuralnet")
library('neuralnet')


set.seed(123)  



# Make predictions on the test data
predictions_nn <- predict(nn_model, newdata = test_pca_data[, -1])

# Convert predictions to binary (1 or 0)
y_pred_nn <- ifelse(predictions_nn > 0.5, 1, 0)

# Evaluate the neural network model
confusionMatrix(y_pred_nn, test_pca_data$diagnosis, positive = "1")

######################



