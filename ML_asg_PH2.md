---
title: "Practical Machine Learning Assignment - Quantified Self Data"
author: "Paul Hodgson"
date: "10 October 2015"
output: html_document
---

This code describes the process of building a machine learning algorythm to predict activity quality.

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: 
Class A: exactly according to the specification
Class B: throwing the elbows to the front 
Class C: lifting the dumbbell only halfway 
Class D: lowering the dumbbell only halfway 
Class D: throwing the hips to the front

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. They are in a field named "classe"

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3oBHWopGr

The task is to build a model which correctly predicts the class based on the other factors.


#Load and prepare the data


```r
setwd("C://Work//Data_Science//Machine_Learning//assg")
```
##Describe the data and its source

The data consists of 19622 observations of 160 variables. The key field is the last 'classe' which contains the 5 classes A-E.

There appear to be lots of missing data and also "#DIV/0!" values and so we'll reload it, marking these as na.


```r
pml_train = read.table("pml-training.csv", 
                               header = TRUE, sep = ",", 
                              na.strings = c("NA", "#DIV/0!"))
```

```
## Warning in file(file, "rt"): cannot open file 'pml-training.csv': No such
## file or directory
```

```
## Error in file(file, "rt"): cannot open the connection
```

##Minimise the number of fields

The first fields contain information about the participants and the timestamps. As they are not sensor readings and we won't be using them as predictors, we can remove them.


```r
sensor_train <- subset(pml_train, select=-c(1:7))#create a new data set without the first 7 fields. 
```

```
## Error in subset(pml_train, select = -c(1:7)): object 'pml_train' not found
```

Some of the fields also appear to have a large number of missing values. To quantify this, we can create a new data frame that counts these.


```r
count_na <- is.na(sensor_train) #create a matrix of TRUE/FALSE for the na values
```

```
## Error in eval(expr, envir, enclos): object 'sensor_train' not found
```

```r
keep_sensor_train <- colSums(count_na) #make a count of the number of times TRUE occurs in each field
```

```
## Error in is.data.frame(x): object 'count_na' not found
```

```r
keep_sensor_train <- data.frame(keep_sensor_train) #format as a data frame
```

```
## Error in data.frame(keep_sensor_train): object 'keep_sensor_train' not found
```

```r
library(plyr)
count(keep_sensor_train, "keep_sensor_train") #produce a summary table of frequencies of na counts
```

```
## Error in count(keep_sensor_train, "keep_sensor_train"): object 'keep_sensor_train' not found
```

There is a clear jump between 53 of the fields which have no missing values and the other 100 fields which have over 19,215 missing values (98% or more). We can use the new data frame to subset just the fields with complete data that we want to keep - and classe.


```r
keep <- which(colSums(count_na) < 19216) #identify fields with less than the agreed number of na s
```

```
## Error in is.data.frame(x): object 'count_na' not found
```

```r
sensor_train_data <- sensor_train[ , keep] #create a new dataset with just the fields that we want to use on our model
```

```
## Error in eval(expr, envir, enclos): object 'sensor_train' not found
```
Check: this creates a dataframe with 19622 observations of 52 fields + classe


#Create training and testing datasets

The question asks us to estimate the out of sample error and so we need to create a training data subset from the our subset of the "pml_train" dataset (NOTE: this "testing" dataset should not be confused with the "pml_test" dataset). This will allow us to test the model independently of the final test data.


```r
library(kernlab); library(caret)
inTrain <- createDataPartition(y=sensor_train_data$classe, p=0.75, list=FALSE) #use 75% / 25% split
```

```
## Error in createDataPartition(y = sensor_train_data$classe, p = 0.75, list = FALSE): object 'sensor_train_data' not found
```

```r
training <- sensor_train_data[inTrain,] #use 75% of observations for training
```

```
## Error in eval(expr, envir, enclos): object 'sensor_train_data' not found
```

```r
testing <- sensor_train_data[-inTrain,] #use 25% of observations for out of sample error testing
```

```
## Error in eval(expr, envir, enclos): object 'sensor_train_data' not found
```

```r
dim(training) #check the data split has worked as expected
```

```
## [1] 251 131
```

```r
dim(testing)
```

```
## [1]  82 131
```

#Choose Model

Linear regression is one of the simplest form of model and so is often worth trying first. However, the 'glm' option in the 'train' function can only be used with 2-class outcomes and so is not suitable for our task. One of the most accurate types of model is Random Forest and so we will try this next, with classe as our outcome and the 52 variables as predictors. As the number of trees increases, so does accuracy, however, so does run time. As the sample size isn't too large and a multi-core PC is available, we will run with a relatively high number of tress, 100.

```r
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(pROC)
```

```
## Type 'citation("pROC")' for a citation.
## 
## Attaching package: 'pROC'
## 
## The following objects are masked from 'package:stats':
## 
##     cov, smooth, var
```

```r
rf <- randomForest(classe ~ ., data = training, ntree = 100)
```

```
## Error in eval(expr, envir, enclos): object 'classe' not found
```

```r
rf # print stats for our model
```

```
## function (n, df1, df2, ncp) 
## {
##     if (missing(ncp)) 
##         .Call(C_rf, n, df1, df2)
##     else (rchisq(n, df1, ncp = ncp)/df1)/(rchisq(n, df2)/df2)
## }
## <bytecode: 0x00000000169ae798>
## <environment: namespace:stats>
```

The confusion matrix shows that our predictions are correct for most cases in the training data (14,639 out of 14,718). Giving an error of 0.54%. This coresponds to the estimated 'Out of Box' error.

We now test our predictions against the testing data using the predict function.


```r
pred <- predict(rf,testing); testing$predRight <- pred==testing$classe #run the model on the testing dataset #create an extra field to record the matches
```

```
## Error in UseMethod("predict"): no applicable method for 'predict' applied to an object of class "function"
```

```
## Error in eval(expr, envir, enclos): object 'pred' not found
```

```r
table(pred,testing$classe) #compare our predictions
```

```
## Error in table(pred, testing$classe): object 'pred' not found
```
The table compares our predictions against the actual values for the data that we held back for cross validation purposes. Although our model has missed a few values, nearly all have been correctly predicted (4,882 out of 4904) giving an error for this testing dataset of 0.45%.

#Test our model on the supplied 20 observations

```r
pml_test = read.table("pml-testing.csv", 
                               header = TRUE, sep = ",", 
                              na.strings = c("NA", "#DIV/0!"))
```

```
## Warning in file(file, "rt"): cannot open file 'pml-testing.csv': No such
## file or directory
```

```
## Error in file(file, "rt"): cannot open the connection
```

```r
ans <- predict(rf,pml_test) #creates a factor with 20 values using our model rf
```

```
## Error in UseMethod("predict"): no applicable method for 'predict' applied to an object of class "function"
```

##export 20 files for uploading
The final step is to create 20 individual files (each with the coreect ID value and just a single letter answer). This is done with the code provided by the tutors for the purpose.


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

setwd("D://Data_Science//for_clone//assg//upload_files")

pml_write_files(ans)
```

```
## Error in pml_write_files(ans): object 'ans' not found
```


