# Set up work env and load libraries
Sys.setenv(language="en")

if(!require("pacman")) install.packages("pacman")
library("pacman")
p_load("ggplot2")
p_load("knitr")
p_load("caret")
p_load("raster")
p_load("scales")
p_load("Boruta")
p_load("randomForest")
p_load("DMwR")
p_load("e1071")
p_load("ROSE")
p_load("ROCR")
p_load("dplyr")
p_load("corrplot")
p_load("RCurl")
p_load("Matrix") 
p_load("stats")
p_load("car")
p_load("MASS")
p_load("fBasics")
p_load("reshape2")
p_load("data.table")

# Overview of SECOM dataset
# Getting SECOM dataset
secom.data<-read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data")
secom.label<-read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data")
colnames(secom.data)<-paste("Feature", 1:ncol(secom.data), sep = "_")
colnames(secom.label)<-c("Status", "Timestamp")
secom<-cbind(secom.label,secom.data)
kable(head(secom[,1:8],15))
dim(secom.data)

# The SECOM dataset includes 1567 rows (observations) with 590 columns representing 590 features/signals collected from sensors, together with the labels representing pass (-1) / fail (1) yield for in house line testing and associated date time stamp.  
# Challenges of SECOM dataset and Data Preparation Steps
## 1. Split the dataset into Training and Test set
# data frame of Frequency of Pass and Fail
secom.status<-data.frame(table(secom$Status,dnn = c("Status")))

# Bar chart of Frequency of Pass and Fail
par(las=2)
secom.barplot.1<-barplot(table(secom$Status),horiz = TRUE,names.arg = c("Pass","Fail"), col = c("limegreen","azure3"), xlim = c(0,1600),main = "Frequency of Pass and Fail")
text(secom.barplot.1,x = table(secom$Status),labels = table(secom$Status), pos = 4)
secom.barplot.1

# Split the dataset with respect to class variables proportions (ratio 14:1)
## generates indexes for randomly splitting the data into training and test sets
set.seed(15)
secom.train_index<-createDataPartition(secom$Status, times = 1,p = 0.8, list = FALSE) # to put 80% of data to training set

## define the training and test sets by using above index
secom.training<-secom[secom.train_index,]
secom.data.train<-secom.training[,-c(1,2)]
secom.test<-secom[-secom.train_index,]

## check the ratio of Pass and Fail in each subset (~ 14:1)
nrow(secom.training[secom.training$Status==-1,])/nrow(secom.training[secom.training$Status==1,])
nrow(secom.test[secom.test$Status==-1,])/nrow(secom.test[secom.test$Status==1,])

## 2. Features with significant Numbers of NA
# function to calculate frequency & percentage of NA
#== na.count [Input: data frame df, dim = 1 (row) or 2 (column) / Output: sum of NA]
secom.na.count <- function(df,dim)                 
{
  apply(df,dim,function(x) sum(is.na(x)))
}

#== na.perc [Input: data frame df, dim = 1 (row) or 2 (column) / Output: percentage of NA]
secom.na.perc <- function(df,dim)
{
  apply(df,dim,function(x) sum(is.na(x))/length(x)*100)
}

# count and percentage of NA in each column
secom.NA.per.col.count<-secom.na.count(secom.data.train,2)
secom.NA.per.col.perc<-secom.na.perc(secom.data.train,2)
secom.NA.col.table<-data.frame(Feature = 1:ncol(secom.data.train), Percentage = round(secom.NA.per.col.perc,3), Frequency = secom.NA.per.col.count)

# Define NA threshold to remove columns
secom.NA.threshold<-20

# plot of Percentage of NA per column
secom.plot.2<-ggplot(secom.NA.col.table,aes(x=Feature,y=Percentage)) + geom_point(aes(color=Frequency), size = 2) + scale_color_continuous(low = "limegreen", high = "dodgerblue4") + theme_bw() + ggtitle("Percentage of NaN per Feature") + theme(plot.title = element_text(hjust = 0.5,face = "bold"),axis.text = element_text(size = 10, color = "black"))

# plot the line of threshold
secom.plot.2<-secom.plot.2+geom_line(aes(x=Feature, y = secom.NA.threshold), col="red") + geom_line(aes(x=Feature, y= 10), col="darkblue", linetype="dashed")
secom.plot.2

# histogram of Percentage of NA per column
secom.plot.3<-ggplot(secom.NA.col.table) + geom_histogram(aes(x = Percentage), binwidth = 10, boundary = 0, closed = "left", fill="limegreen", col = "black") + scale_x_continuous(breaks = seq(0,100,10)) + theme_bw() + ggtitle("Percentage of NaN per Feature") + theme(plot.title = element_text(hjust = 0.5,face = "bold"),axis.text = element_text(color = "black")) + ylab("Number of features") + xlab("Percentage of NA")
# plot the NA threshold line
secom.NA.line<-data.frame(secom.NA.line=secom.NA.threshold)
secom.plot.3 + geom_vline(data = secom.NA.line, aes(xintercept=secom.NA.line), linetype = "dashed", color="red")

# Looking at the percentage of "NaN" per column, I determined our threshold to __20%__ (red line). In other words, any column that has greater than 20% of "NaN" shall be removed.

# Dataset after removing columns with greater than 20% of NA 
secom.clean.1<-secom.data.train[, ! secom.na.perc(secom.data.train,2)>secom.NA.threshold]

## 3. Outliers
# standardize (z-score) secom
secom.zscore<-data.frame(scale(secom.clean.1))

# count of outliers in each column using 3s rule
secom.outliers<-apply(secom.zscore,2,function(x) length(which(abs(x)>3)))

# I considered whether the entire record could be removed, hence looked at the at the percentage of "NaN" per row.
# count and percentage of NA per row
secom.data.pass<-secom.training[secom.training$Status==-1,3:ncol(secom.training)]  # data frame of features of only Pass cases
secom.NA.per.row.perc <- secom.na.perc(secom.data.pass,1)
secom.NA.per.row.count<- secom.na.count(secom.data.pass,1)
secom.NA.row.table<-data.frame(Row = 1:nrow(secom.training[secom.training$Status ==-1,]), Percentage = round(secom.NA.per.row.perc,3), Frequency = secom.NA.per.row.count)

# plot of Percentage of NA per row
secom.plot.4<-ggplot(secom.NA.row.table,aes(x=Row,y=Percentage)) + geom_point(aes(color=Frequency), size = 2) + scale_color_continuous(low = "limegreen", high = "dodgerblue4") + theme_bw() + ggtitle("Percentage of NaN per Record") + theme(plot.title = element_text(hjust = 0.5,face = "bold"),axis.text = element_text(size = 10, color = "black"))
secom.plot.4

# histogram of Percentage of NA per row
secom.plot.5<-ggplot(secom.NA.row.table) + geom_histogram(aes(x = Percentage), binwidth = 2, boundary = 0, closed = "left", fill="limegreen", col = "black") + scale_x_continuous(breaks = seq(0,30,2)) + theme_bw() + ylab("Number of features") + xlab("Percentage of NA") + ggtitle("Percentage of NaN per Record") + theme(plot.title = element_text(hjust = 0.5,face = "bold"),axis.text = element_text(color = "black"))
secom.plot.5

# The percentage of NA per row would be too low to exclude entire records
# Therefore, transform the outlier to NA and impute values to replace outliers in the following stages.

# Transform outliers to NA
secom.trimmed<-secom.clean.1
secom.trimmed[abs(secom.zscore)>3] = NA

## 4. Features with low volatility
# SECOM dataset consists of columns with low volatility, which do not show much difference between Pass/Fail cases 
# Coefficient of variance of features 
secom.coef.var<-apply(secom.trimmed,2,cv,na.rm=TRUE)

# Histogram of volatility (The range of absolute coefficient of variance is large but focus on xlim from 0 to 500, because higher volatility columns will be kept anyways)
par(las=1)
hist(abs(secom.coef.var),breaks=seq(0,50000,5), xlim=c(0,500), col = "limegreen", main = "Histogram of volatility", xlab = "CV", ylab = "Number of features", cex.main=1.3)
abline(v=5, col = "red", lty = 2)

# volatility threshold = 5 and will remove the columns with volatility less than threshold
# volatility threshold
secom.vola.threshold<-5
# Dataset after removing low-volatility features
secom.clean.2<-secom.trimmed[,which(abs(secom.coef.var)>secom.vola.threshold)]

#== Bar chart: change of number of Features
secom.clean<-c(ncol(secom.data.train),ncol(secom.clean.1),ncol(secom.clean.2))
secom.plot.6<-barplot(secom.clean, names.arg = c("Original data","After removing \n high NA","After removing \n low volatility"),ylab = "Number of features", col = "limegreen",ylim = c(0,650),main = "Change of Number of Features")
text(secom.plot.6, y=secom.clean, secom.clean, pos = 3)

## 5. Data Scaling/ Transformation
# The features of SECOM dataset have different unit of measures and their range of values vary widely
# therefore scaling is necessary especially when some algorithms take the distance between data points into consideration.
# Min-Max scaling of range [0,1]
secom.scaled<-data.frame(apply(secom.clean.2,2,rescale,to=c(0,1)))

# Algorithms with assumption of normal distribution will require data transformation before building the model. 
# Hence, a Yeo-Johnson transformation function is prepared for later step if necessary (because SECOM contains zero and negative values).
secom.yj.transform<-function(original.var){
  
  # determine best lambda for Yeo-Johnson
  yj.best.power<-powerTransform(original.var, family="yjPower")
  
  # transform original variable with Yeo-Johnson
  yj.best.data<-yjPower(original.var,yj.best.power$lambda)
  
}

## 6. Imputation
### 6b. kNN Imputation
# kNN approximates the missing value based on the its neighboring values. 
# While it is simple, it gives a more accurate approximation compared to mean imputation. One drawback to using it is the long computation time for large datasets. 

# kNN Imputation
secom.knn.imputed<-knnImputation(data = secom.scaled,k = 5, scale = F)
# Check whether all NA have been imputed
anyNA(secom.knn.imputed)

## 7. Feature selection
### 7b. Principal Component Analysis (PCA)
# PCA as a method for feature selection/reduction. 
# Transform variables to components
secom.pca.comp <- prcomp(secom.knn.imputed)
# std deviation of each component
secom.std.comp <- secom.pca.comp$sdev
# variance of the component
secom.var.comp <- secom.std.comp^2
# proportion of variance explained by the components
secom.var.explained <- secom.var.comp/sum(secom.var.comp)*100
# Scree plot
# Based on the scree plot, the curve is smooth and there is no clear break (elbow) that can be seen.
plot(secom.var.explained, xlab = "Principal Component", ylab = "Proportion of Variance Explained", type = "b", col="limegreen", main="Scree plot")

# cumulative scree plot of PCs & variance explained
# looking at the cumulative percentage of the variance explained, 180 PCs explain 98% the proportion of variance, with the highest PC accounting for only 4%. Hence, PCA is not a good method in this case.
plot(cumsum(secom.var.explained), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", type = "b", col="limegreen", main="Cummulative Proportion of Variance Explained by Principal Components")

### 7c. Boruta
# is a wrapper built around the random forest classification algorithm to give a numerical estimate of the feature importance. 
# It creates shadow features and using the random forest classifier, compares the z-scores of the shadow and actual features in a number of iterations. 
# It then selects the features that performed better than the shadow features.

# add back the Status to the dataset
secom.knn.imputed$Status<-secom.training$Status
# build Boruta function [Input: a data frame / Output: features selected by Boruta]
secom.my.boruta<-function(df){  
  # run Boruta
  set.seed(2000)
  secom.Boruta<<-Boruta(Status~.,data = df, doTrace = 2, maxRuns = 300)
  # Boruta result
  plot(secom.Boruta)
  select<-getSelectedAttributes(TentativeRoughFix(secom.Boruta))
}

# get selected features
secom.selected.features<-secom.my.boruta(secom.knn.imputed)

# The most important features selected by Boruta are as follows:
print("The most important features selected by Boruta")
print(secom.selected.features)

# create training set for model with selected features
secom.train.model<-secom.knn.imputed[,c("Status",secom.selected.features)]
# ensure Status defined as Factor
secom.train.model$Status<-as.factor(secom.train.model$Status)

#== Bar chart: change of number of Features
secom.clean<-c(ncol(secom.data.train),ncol(secom.clean.1),ncol(secom.clean.2), ncol(secom.train.model[,-1]))
secom.plot.7<-barplot(secom.clean, names.arg = c("Original data","After removing \n high NA","After removing \n low volatility", "After feature \n selection"),ylab = "Number of features", col = "limegreen",ylim = c(0,650),main = "Change of Number of Features", cex.main=1.3)
text(secom.plot.7, y=secom.clean, secom.clean, pos = 3)

## 8. Balancing the dataset
# Most machine learning classification algorithms are sensitive to imbalance in the classes. An imbalance dataset will bias the predictive model towards the majority class. 
# Therefore, need to implement a certain methods to balance the data in order to build an efficient model.
# ROSE
secom.train.model.rose<-ROSE(Status~.,secom.train.model,seed = 1)$data
print("Number of each class after ROSE")
print(table(secom.train.model.rose$Status))

## 9. Model building
### 9a. Build a simple Random Forest 
# Build a simple Random Forest model
set.seed(2000)
secom.model_rf.rose<-randomForest(Status~.,secom.train.model.rose,importance = TRUE, mtry =5)

print("Random Forest model built on ROSE")
print(secom.model_rf.rose)

### 9b. Predict on Test set
# Prepare Test set with features selected in Training set
secom.test.model<-secom.test[,c("Status", secom.selected.features)]

# [0,1] Min-max scaling for Test set
secom.test.model.rescaled<-data.frame(Status = secom.test.model[,1],apply(secom.test.model[,-1],2,rescale,to=c(0,1)))

# remove rows that have NA value to avoid error in next prediction step
secom.test.model.rescaled<-secom.test.model.rescaled[rowSums(is.na(secom.test.model.rescaled)) == 0,]

### 9c. Assess the model performance
# Assess the model performance
## predict function
secom.my.predict<-function(model,testset, threshold=0.5){
  # Predict using model built on ROSE
  model_predict<-predict(model,testset,type="prob")
  # data frame of actual Status and predicted Status
  final<-data.frame(actual = testset$Status%>%as.factor(), predict = model_predict)
  # 
  final$predict<-ifelse(final$predict.1>threshold, 1, -1)%>%as.factor()
  # Confusion Matrix
  cm<-confusionMatrix(final$predict,final$actual,positive = "1")
  # ROC Curve 
  pre<-prediction(final$predict.1, final$actual)
  perf<-performance(pre,measure = "tpr", x.measure = "fpr")
  plot(perf, col="red")
  # AUC
  auc<-performance(pre, "auc")
  auc<-auc@y.values[[1]]
  legend(x = 0.5, y= 0.3,legend = round(auc,3), title = "AUC")
  abline()
  # list combining all the results above
  result<-list(overall=cm$overall[c("Accuracy","Kappa")],byClass=cm$byClass, conf.matrix=cm$table, AUC = auc)
}

# predict and assess model performance (default threshold = 0.5)
secom.predict<-secom.my.predict(secom.model_rf.rose, secom.test.model.rescaled)
print(secom.predict)

# assess Sensitivity and Specificity across different theshold
secom.prob.thres<-seq(0.1,0.5,0.01)
Specificity<-numeric(length(secom.prob.thres))
Sensitivity<-numeric(length(secom.prob.thres))
for (i in seq(1:length(secom.prob.thres))){
  Specificity[i]<-secom.my.predict(secom.model_rf.rose,secom.test.model.rescaled, secom.prob.thres[i])$byClass["Specificity"]
  Sensitivity[i]<-secom.my.predict(secom.model_rf.rose,secom.test.model.rescaled, secom.prob.thres[i])$byClass["Sensitivity"]
}
secom.compare<-data.frame(Threshold = secom.prob.thres, Sensitivity = Sensitivity, Specificity = Specificity)
secom.compare<-melt(secom.compare, id.vars = "Threshold", variable.name="Measures", value.name="Data")
ggplot(secom.compare,aes(x=Threshold, y=Data, color=Measures)) + geom_line() + ylab("") + ggtitle("Sensitivity & Specificity across thresholds") + theme(plot.title = element_text(hjust = 0.5,face = "bold"),axis.text = element_text(size = 12), axis.title = element_text(size = 13), legend.text = element_text(size = 13), legend.title = element_text(size = 13))

# adjust the probability threshold to 0.3 in order to improve sensitivity
secom.predict.30<-secom.my.predict(secom.model_rf.rose, secom.test.model.rescaled, 0.3)
secom.predict.30

