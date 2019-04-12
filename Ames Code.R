# Loading libraries
library(stringr)
library(gdata)
library(tidyverse)
library(plyr)
library(lubridate)
library(psych)
library(scales)
library(graphics)
library(car)
library(onehot)
library(outliers)
library(robustHD)
library(ggplot2)
library(robust)
library(caret)
library(corrplot)
library(ggpubr)
library(moments)
library(dummies)
library(glmnet)
library(xgboost)
library(randomForest)

# Loading the dataset
data = read.csv("Ames_data.csv")
head(data)
dim(data)

# Missing values
missing.n = sapply(names(data), function(x) length(which(is.na(data[, x]))))
missing.n
which(missing.n > 0) #Shows that the 60th feature: Garage_Yr_Blt has missing values
summary(data)
id = which(is.na(data$Garage_Yr_Blt))
length(id)
data$Garage_Yr_Blt[id] <- 0 #Replacing the NA values by 0


# Spitting the data into Train and Test sets (There are total 10 sets of train and test which can be generated from the ID's provided in the below R file)
load("project1_testIDs.R")
j <- 10
test.dat <- data[testIDs[,j], ]
train.dat <- data[-testIDs[,j], ]
PID = test.dat$PID

# Removing the first IDs and the last column target
train.y <- log(train.dat$Sale_Price)
train.x <- subset(train.dat, select = -c(PID, Sale_Price))
test.y <- log(test.dat$Sale_Price)
test.x <- subset(test.dat, select = -c(PID, Sale_Price))

# Convert Month Sold and Year Sold to factors(levels)
train.x$Mo_Sold = as.factor(train.x$Mo_Sold)
test.x$Mo_Sold = as.factor(test.x$Mo_Sold)
train.x$Year_Sold = as.factor(train.x$Year_Sold)
test.x$Year_Sold = as.factor(test.x$Year_Sold)

# Random Forest
set.seed(100)
rf_classifier = randomForest((train.y) ~ ., data=train.x, ntree=500, mtry=100, importance=TRUE)
Ytest.pred.rf <- predict(rf_classifier,test.x)
error1 = sqrt(mean((Ytest.pred.rf - (test.y))^2))
error1

#output1 = data.frame(PID, exp(Ytest.pred.rf))
#colnames(output1) = c('PID','Sale_Price')
#write.csv(output1,'mysubmission1.txt',row.names = FALSE)


# Removing a few variables which are highly imbalanced
train.x <- subset(train.x, select = -c(Street, Utilities, Condition_2, Roof_Matl, Heating,
                                       Pool_QC, Misc_Feature , Low_Qual_Fin_SF, Pool_Area, 
                                       Longitude, Latitude))
test.x <- subset(test.x, select = -c(Street, Utilities, Condition_2, Roof_Matl, Heating,
                                      Pool_QC, Misc_Feature , Low_Qual_Fin_SF, Pool_Area, 
                                      Longitude, Latitude))

# Winsorization
train.x$Lot_Frontage = winsorize(train.x$Lot_Frontage, minval = NULL)
train.x$Lot_Area = winsorize(train.x$Lot_Area, minval = NULL)
train.x$Mas_Vnr_Area = winsorize(train.x$Mas_Vnr_Area, minval = NULL)
train.x$BsmtFin_SF_2 = winsorize(train.x$BsmtFin_SF_2, minval = NULL)
train.x$Bsmt_Unf_SF = winsorize(train.x$Bsmt_Unf_SF, minval = NULL)
train.x$Total_Bsmt_SF = winsorize(train.x$Total_Bsmt_SF, minval = NULL)
train.x$Second_Flr_SF = winsorize(train.x$Second_Flr_SF, minval = NULL)
train.x$First_Flr_SF = winsorize(train.x$First_Flr_SF, minval = NULL)
train.x$Gr_Liv_Area = winsorize(train.x$Gr_Liv_Area, minval = NULL)
train.x$Garage_Area = winsorize(train.x$Garage_Area, minval = NULL)
train.x$Wood_Deck_SF = winsorize(train.x$Wood_Deck_SF, minval = NULL)
train.x$Open_Porch_SF = winsorize(train.x$Open_Porch_SF, minval = NULL)
train.x$Enclosed_Porch = winsorize(train.x$Enclosed_Porch, minval = NULL)
train.x$Three_season_porch = winsorize(train.x$Three_season_porch, minval = NULL)
train.x$Screen_Porch = winsorize(train.x$Screen_Porch, minval = NULL)
train.x$Misc_Val = winsorize(train.x$Misc_Val, minval = NULL)

test.x$Lot_Frontage = winsorize(test.x$Lot_Frontage, minval = NULL)
test.x$Lot_Area = winsorize(test.x$Lot_Area, minval = NULL)
test.x$Mas_Vnr_Area = winsorize(test.x$Mas_Vnr_Area, minval = NULL)
test.x$BsmtFin_SF_2 = winsorize(test.x$BsmtFin_SF_2, minval = NULL)
test.x$Bsmt_Unf_SF = winsorize(test.x$Bsmt_Unf_SF, minval = NULL)
test.x$Total_Bsmt_SF = winsorize(test.x$Total_Bsmt_SF, minval = NULL)
test.x$Second_Flr_SF = winsorize(test.x$Second_Flr_SF, minval = NULL)
test.x$First_Flr_SF = winsorize(test.x$First_Flr_SF, minval = NULL)
test.x$Gr_Liv_Area = winsorize(test.x$Gr_Liv_Area, minval = NULL)
test.x$Garage_Area = winsorize(test.x$Garage_Area, minval = NULL)
test.x$Wood_Deck_SF = winsorize(test.x$Wood_Deck_SF, minval = NULL)
test.x$Open_Porch_SF = winsorize(test.x$Open_Porch_SF, minval = NULL)
test.x$Enclosed_Porch = winsorize(test.x$Enclosed_Porch, minval = NULL)
test.x$Three_season_porch = winsorize(test.x$Three_season_porch, minval = NULL)
test.x$Screen_Porch = winsorize(test.x$Screen_Porch, minval = NULL)
test.x$Misc_Val = winsorize(test.x$Misc_Val, minval = NULL)

# PreProcessing Matrix Output
PreProcessingMatrixOutput <- function(train.data, test.data){
  # generate numerical matrix of the train/test
  # assume train.data, test.data have the same columns
  categorical.vars <- colnames(train.data)[which(sapply(train.data, 
                                                        function(x) is.factor(x)))]
  train.matrix <- train.data[, !colnames(train.data) %in% categorical.vars, drop=FALSE]
  test.matrix <- test.data[, !colnames(test.data) %in% categorical.vars, drop=FALSE]
  n.train <- nrow(train.data)
  n.test <- nrow(test.data)
  for(var in categorical.vars){
    mylevels <- sort(unique(train.data[, var]))
    m <- length(mylevels)
    tmp.train <- matrix(0, n.train, m)
    tmp.test <- matrix(0, n.test, m)
    col.names <- NULL
    for(j in 1:m){
      tmp.train[train.data[, var]==mylevels[j], j] <- 1
      tmp.test[test.data[, var]==mylevels[j], j] <- 1
      col.names <- c(col.names, paste(var, '__', mylevels[j], sep=''))
    }
    colnames(tmp.train) <- col.names
    colnames(tmp.test) <- col.names
    train.matrix <- cbind(train.matrix, tmp.train)
    test.matrix <- cbind(test.matrix, tmp.test)
  }
  return(list(train = as.matrix(train.matrix), test = as.matrix(test.matrix)))
}

r = PreProcessingMatrixOutput(train.x, test.x)
train.x = r$train
test.x = r$test

train.x <- subset(train.x, select = -c(Mas_Vnr_Area, BsmtFin_SF_2, Second_Flr_SF, Wood_Deck_SF, Enclosed_Porch, Three_season_porch, Screen_Porch, Misc_Val))
test.x <- subset(test.x, select = -c(Mas_Vnr_Area, BsmtFin_SF_2, Second_Flr_SF, Wood_Deck_SF, Enclosed_Porch, Three_season_porch, Screen_Porch, Misc_Val))

# LASSO
set.seed(100)
cv.out <- cv.glmnet(as.matrix(train.x), as.matrix(train.y), alpha = 1)
Ytest.pred.lasso <-predict(cv.out, s = cv.out$lambda.min, newx = test.x)
error2 = sqrt(mean((Ytest.pred.lasso - (test.y))^2))
error1
error2

#output2 = data.frame(PID, exp(Ytest.pred.lasso))
#colnames(output2) = c('PID','Sale_Price')
#write.csv(output2,'mysubmission2.txt',row.names = FALSE)

