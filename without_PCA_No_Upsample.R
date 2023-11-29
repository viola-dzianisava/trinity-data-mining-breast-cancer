install.packages("dummies")
library(class)
library(gains)
library(rpart)
library(caret)
library(randomForest)
library(dummies)
install.packages("rpart")
library(caret)
library(rpart)
library(rpart.plot)

Breastcancer <- read.csv("~/Desktop/data_c.csv")
View(Breastcancer)

#drop columns
Breastcancer <- Breastcancer[, -1] # dropping columns 1 

#turn all values into 1 for M, 0 for B 
Breastcancer$diagnosis <- ifelse(Breastcancer$diagnosis == 'M', 1, 0)

#partitioning the data

# For classification, we need to factorize! 

Breastcancer$diagnosis <- as.factor(Breastcancer$diagnosis)
lapply(Breastcancer, class) 

# Splitting data
set.seed(123)

train.index <- sample(c(1:dim(Breastcancer)[1]), dim(Breastcancer)[1]*0.6)
valid.index <- setdiff(c(1:dim(Breastcancer)[1]), train.index)

length(train.index)
length(valid.index)

train.df <- Breastcancer[train.index,]
valid.df <- Breastcancer[valid.index,]

# Normalizing the data
norm.values <- preProcess(train.df, method = "range")
train.norm.df <- predict(norm.values, train.df)
valid.norm.df <- predict(norm.values, valid.df)

# building the models
# logistic regression
reg <- glm(train.df$diagnosis ~ ., data= train.df, family= "binomial")
summary(reg)
pred <- ifelse(predict(reg, valid.df, type= "response") > 0.5,1,0)
Cm <- confusionMatrix(as.factor(pred), valid.df$diagnosis, positive="1") 
Cm


#KNN Model
#8 is my target variable, cl is the class
Kn <- class::knn(train=train.df[,-1], test= valid.df[,-1], cl= train.df[,1], k=3, prob=TRUE)
Kn

# we get two different output: Binary and Probability ~ class is 0 for a prob of 1 
Cmkn<-confusionMatrix(Kn, valid.df[,1], positive="1")
Cmkn

# decision tree
Tr <- rpart(diagnosis ~., train.df)
Pred <- ifelse(predict(Tr, valid.df)[,2]>0.5,1,0)

# we get 2 column output, we’ll use prob of #classifying things with 1 
Cmtr <- confusionMatrix(as.factor(Pred), factor(valid.df$diagnosis), positive="1")
Cmtr




#Neural nets
Neuralnn_ <- neuralnet(train.norm.df$diagnosis ~ ., data = train.norm.df, linear.output = F, hidden = c(3))

# Make predictions on the validation dataset
predictions_ <- predict(Neuralnn_, newdata = valid.norm.df)
head(predictions_)
predicted_classes_ <- ifelse(predictions > 0.5, 1, 0)  
# Create a confusion matrix
conf_matrix_ <- table(Actual = valid.norm.df$diagnosis, Predicted = predicted_classes_[,2])
confusionMatrix(as.factor(valid.norm.df$diagnosis),as.factor(predicted_classes_[,2]), positive = '1')
# Print the confusion matrix
print(conf_matrix)
head(Neuralnn$data)
Validforneuralnetwork<- valid.norm.df
head(Validforneuralnetwork)
names(Validforneuralnetwork)[names(Validforneuralnetwork)=="concave points_mean"] <- "concave_points_mean"
names(Validforneuralnetwork)[names(Validforneuralnetwork)=="concave points_se"] <- "concave_points_se"

# Model 5: Naive Bayes 

# Naive Bayes model
naive_bayes_model_ <- naiveBayes(diagnosis ~ ., data = train.norm.df)

# Make predictions on the test data
predictions_nb_ <- predict(naive_bayes_model_, newdata = valid.norm.df)
head(predictions_nb_)

# Evaluate the Naive Bayes model
confusionMatrix(predictions_nb_, valid.norm.df$diagnosis, positive= "1") 

