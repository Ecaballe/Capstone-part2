
###############################################################
###########  Capstone Project Accelerometer Data
################################################################



# Load required packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(gtools)) install.packages("gtools", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("ddplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(skimr)) install.packages("skimr", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(data.table)
library(dslabs)
library(broom)
library(gtools)
library(dplyr)
library(ggplot2)
library(lubridate)
library(randomForest)

library(skimr)
library(rpart)
library(ranger)



# Uncomment following line to Load data file from local directory
#path <- "accelerometer.csv"

# Load Data from UCI website

dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00611/accelerometer.csv", dl)

accel <- read.csv(dl, header = TRUE, sep = ",")



##################################################
## wconfid: configuration of the weights on the fan, pctid: percent max speed the motor was run, (x, y, z) vibration levels
##################################################


    
## Calculate Magnitude of vibration

accel <- accel %>% mutate(vibe = sqrt(x^2 + y^2 + z^2), balance = factor(wconfid), pct_rpm = factor(pctid))

## Box Plot of Vibration Magnitudes at various RPM values for balance configuration 1

rpms <- c("20", "25", "30", "35", "40", "45", "50", "75", "100")
accel %>% filter(pct_rpm %in% rpms, balance == "1") %>% ggplot(aes(balance,vibe, fill = pct_rpm)) +
  geom_boxplot() 

## Box Plot of Vibration Magnitudes 
rpms <- c("60", "55", "80", "65", "70", "45", "50", "75", "100")
vibe_plot_rpms <- accel %>% filter(pct_rpm %in% rpms) %>% ggplot(aes(balance,vibe, fill = pct_rpm)) +
  geom_boxplot() 
vibe_plot_rpms


##plot vibration levels vs balance
accel %>% ggplot(aes(z,vibe, color = balance)) + geom_point()

plot(accel$wconfid, accel$vibe)

plot(accel$y, accel$vibe)

boxplot(vibe ~ balance, data = accel )





#####################################
# Create a test and train dataset
#####################################

#Since vibration levels are low when spinning at lower rpms, remove rpms lower than 50% from the data set

accel <- accel %>% filter(pctid >= 50)
str(accel)

#Create a validation data set out of 10% of the data

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
val_index <- createDataPartition(y = accel$wconfid, times = 1, p = 0.1, list = FALSE)
accel_data <- accel[-val_index,]
validation  <- accel[val_index,]



#Split the accel dataset into a 20% test and a 80% train dataset


set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
accel_index <- createDataPartition(y = accel_data$wconfid, times = 1, p = 0.2, list = FALSE)
accel_train <- accel_data[-accel_index,]
accel_test <- accel_data[accel_index,]


sval <- str(validation)
strain <- str(accel_train)
stest <- str(accel_test)


skimmed <- skim(accel_data)
tskim <- skimmed[,1:7]
tskim



# Create smaller samples to test code when the whole data set is too large to compute

set.seed(1, sample.kind="Rounding")
ind <- createDataPartition(y = accel_train$wconfid, times = 1, p = 0.1, list = FALSE)
accel10k <- accel_train[ind,]
str(accel10k)



#########################################################
### Train the model
#########################################################

#Create function to do a Residual Mean Square Error

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#Select method pick predictors (x, y, z, vibe, pct_rpm)

train_set <- accel_train 
str(train_set)

################### knn model with Test_RMSE = 0.6320601

control <- trainControl(method = "cv", number = 10, p = 0.9)
model_knn <- train(wconfid~ vibe + pctid, data = train_set, method = "knn", tuneGrid = data.frame(k = seq(30, 50, 2)), trControl = control)
model_knn

# test model on test set

t_model <-model_knn
rating_hat <- predict(t_model, accel_test)
Test_RMSE <- RMSE(accel_test$wconfid, rating_hat)
Test_RMSE


################### Random Forest with Test_RMSE = 0.6346502

control <- trainControl(method = "cv", number = 10)
grid <- data.frame(mtry = c(1, 5, 10))
model_rf <- train(wconfid ~ vibe + pctid, data = train_set, method = "rf", 
	ntree = 10,
	trControl = control,
	tuneGrid = grid,
	nSamp = 5000)
model_rf

# test model on test set

t_model <-model_rf
rating_hat <- predict(t_model, accel_test)
Test_RMSE <- RMSE(accel_test$wconfid, rating_hat)
Test_RMSE

##################### rpart Test_RMSE =  0.637228


set.seed(1, sample.kind="Rounding")
model_rpart <- train(
  wconfid ~ vibe + pctid, data = train_set, method = "rpart",
  trControl = trainControl("cv", number = 10),
  tuneLength = 20
  )
model_rpart

# Plot model error vs different values of cp (complexity parameter)
plot(model_rpart)

# Print the best tuning parameter cp that minimize the model RMSE
model_rpart$bestTune

# Print the decision tree
plot(model_rpart$finalModel, margin = 0.1)
text(model_rpart$finalModel, cex = 0.75)

# test model on test set

t_model <-model_rpart
rating_hat <- predict(t_model, accel_test)
Test_RMSE <- RMSE(accel_test$wconfid, rating_hat)
Test_RMSE

################### Ranger Test_RMSE =  0.6760426

model_ranger <- train(wconfid ~ vibe + pctid, data = train_set, method = "ranger")
model_ranger

# test model on test set

t_model <-model_ranger
rating_hat <- predict(t_model, accel_test)
Test_RMSE <- RMSE(accel_test$wconfid, rating_hat)
Test_RMSE




###################################################
############# Final Check   knn = 0.6303947, random forest = .6320287
###################################################

f_model <- model_rf
ratingPrediction <- predict(f_model, validation)
Validation_RMSE <- RMSE(validation$wconfid,ratingPrediction)
Validation_RMSE






