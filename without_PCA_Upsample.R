#Without PCA" & Upsampling the data

install.packages('class')
install.packages('gains')
install.packages('rpart')
install.packages('caret')
install.packages('randomForest')
install.packages('dummies')
install.packages('car') # VIF function
library(class)
library(gains)
library(rpart)
library(caret)
library(randomForest)
library(dummies)
library('car')

#Loading the dataset
br_can.df <-read.csv('Breast_Cancer')

#Removing the id column
br_can.df <- br_can.df[, -1]
View(br_can.df)
#Converting the malign to 1 and benign to 0 for the diagnosis column
br_can.df$diagnosis <- ifelse(br_can.df$diagnosis=='M',1,0) 

class(br_can.df$diagnosis) # Numeric

br_can.df$diagnosis <- as.factor(br_can.df$diagnosis)

class(br_can.df$diagnosis) # Factor

lapply(br_can.df, class) # checking for classes for other variables

#Splitting the data 
set.seed(123)

Trainindex <- sample(c(1:dim(br_can.df)[1]), dim(br_can.df)[1]*0.6)
Validindex <- setdiff(c(1:dim(br_can.df)[1]), Trainindex)

length(Trainindex) #341 (60%)
length(Validindex) #228 (40%)

train.df <- br_can.df[Trainindex,] # 0-225 (65%) , 1- 116 (34%)
valid.df <- br_can.df[Validindex,] 

table(train.df$diagnosis) # count of 0s & 1s 

#Normalising the data

norm.values <- preProcess(train.df, method= 'range')
train.norm.df <- predict(norm.values, train.df)
valid.norm.df <- predict(norm.values, valid.df)

View(train.norm.df)
View(valid.norm.df)

table(train.norm.df$diagnosis)
table(valid.norm.df$diagnosis)

# Upsampling the train.norm.df for class balance

ei <- which(colnames(train.norm.df) != "diagnosis")
up_train <- upSample(
  x = train.norm.df[, ei, drop = FALSE],
  y = as.factor(train.norm.df$diagnosis)
)

table(train.norm.df$diagnosis)
table(up_train$Class)

# Changing the name of the dependent variable back to diagnosis
names(up_train)[names(up_train) == "Class"] <- "diagnosis"

# Balanced and normalised dataset
table(up_train$diagnosis) # 225 - 0s, 225 - 1s


# Model 1: Logistic Regression 

Logreg1 <- glm(diagnosis ~., data = up_train, family = 'binomial')

#summary of the model
summary(Logreg1)
install.packages("car")
library(car)
vif(Logreg1) # High Inflation Value which depicts that there is multicollinearity

#Predicting the values from the model for the Validation set 
predlog <- predict(Logreg1, valid.norm.df, type = 'response')

pred <- ifelse(predlog>0.5,1,0)
class(pred)

#Confusion Matrix
#install.packages("yardstick")
#library(yardstick)

ConfusionM_LogReg1 <- confusionMatrix(as.factor(pred), valid.norm.df$diagnosis, positive='1')
ConfusionM_LogReg1
# Model 2:KNN Model with Upsampling

KNN <- class::knn(train = up_train[,-31], test = valid.norm.df[,-1], cl=up_train[,31],k=3, prob = TRUE) 

#Confusion Matrix for KNN
confusionMatrix(KNN, valid.norm.df[,1], positive='1')

# Model 3: Decision Tree with Upsampling

DT <- rpart(diagnosis ~., up_train)

PredDT <- predict(DT, valid.norm.df)

head(PredDT)

PredDTN <- ifelse(PredDT>0.5,1,0)

head(PredDTN)

#Confusion Matrix for DT
confusionMatrix(as.factor(PredDTN[,2]), valid.norm.df$diagnosis, positive="1")

# Create neural network

Neuralnn <- neuralnet(up_train$diagnosis ~ ., data = up_train, linear.output = F, hidden = c(3))

# Make predictions on the validation dataset
predictions <- predict(Neuralnn, newdata = Validforneuralnetwork)
head(predictions)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)  
# Create a confusion matrix
conf_matrix <- table(Actual = Validforneuralnetwork$diagnosis, Predicted = predicted_classes[,2])
confusionMatrix(as.factor(Validforneuralnetwork$diagnosis),as.factor(predicted_classes[,2]), positive = '1')
# Print the confusion matrix
print(conf_matrix)
head(Neuralnn$data)
Validforneuralnetwork<- valid.norm.df
head(Validforneuralnetwork)
names(Validforneuralnetwork)[names(Validforneuralnetwork)=="concave points_mean"] <- "concave_points_mean"
names(Validforneuralnetwork)[names(Validforneuralnetwork)=="concave points_se"] <- "concave_points_se"

# Model 5: Naive Bayes 

# Naive Bayes model
naive_bayes_model <- naiveBayes(diagnosis ~ ., data = up_train)

# Make predictions on the test data
predictions_nb <- predict(naive_bayes_model, newdata = Validforneuralnetwork)
head(predictions_nb)

# Evaluate the Naive Bayes model
confusionMatrix(predictions_nb, Validforneuralnetwork$diagnosis, positive= "1") 
