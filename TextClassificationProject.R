######
# Please change the paths to suit your system
# Directory path
path <- "C://Users//USER//Desktop//notes//CS6405//Newsgroups-1//Newsgroups//"
# Output for Raw Data
output <- "C://Users//USER//Desktop//notes//CS6405//sd.csv"
# Output for CleanData
cleanoutput <- "C://Users//USER//Desktop//notes//CS6405//cleansd.csv"
# Top 200 Frequency Table
top200Path <- "C://Users//USER//Desktop//notes//CS6405//top200.csv"
######
# Code to Create a dataframe with unique words and its frequencies
result <- dataFrame(path, output, 1)
wordsFrequency <- result[[2]]
word_frequency_table <- data.frame(table(wordsFrequency))
word_frequency_table <- word_frequency_table[order(-word_frequency_table$Freq),]
word_frequency_table <- word_frequency_table[c(1:200),]
write.csv(word_frequency_table, top200Path)
print(result[[1]])
######

# Create Bag of words on the raw data set:
set.seed(4060)
dat <- read.csv(output)
df <- popRows(dat)
nrow(df)
df <- df[-c(1:2),]
df <- data.frame(df)
df <- japply(df, which(sapply(df, class)=="factor"), as.character)
df <- japply(df, which(sapply(df, class)=="character"), as.integer)
# Divide df i.e. bag of words into training and testing set
set.seed(4060)
s <- sample(1:nrow(df),round(0.7*nrow(df)))
training <- df[s,]
testing <- df[-s,]
######

#naive bayes
y.hat.index <- 1
y.hat <- c()
words <- colnames(df)
for(f in 1:nrow(testing)){
  y.hat[y.hat.index] <- naiveBayes(testing[f,], words, training)
  y.hat.index <- y.hat.index + 1
}
tb.nb <- table(y.hat, testing[,which(colnames(testing) == "folderTypeClass")])
# Accuracy of Naive Bayes
sum(diag(tb.nb))/sum(tb.nb)

# Data divison for KNN and Random Forest.
tx <- df[,-which(colnames(training) == "folderTypeClass")]
ty <- df[,which(colnames(training) == "folderTypeClass")]
testx <- df[,-which(colnames(testing) == "folderTypeClass")]
testy <- df[,which(colnames(testing) == "folderTypeClass")]

# KNN
set.seed(4060)
library(class)
k <- knn(train = tx, test = testx, cl = ty, k = 2)
tb <- table(testy, k)
# Accuracy of KNN
sum(diag(tb))/sum(tb)

# Random Forest
set.seed(4060)
library(randomForest)
ty <- as.factor(ty)
rf.out <- randomForest(tx,ty, ntree = 10)
names(rf.out)
rf.pred <- predict(rf.out,newdata = testx)
tb.rf <- table(testy, rf.pred)
# Accuracy of Random Forest
sum(diag(tb.rf))/sum(tb.rf)

# clean data set
# Remove punctuations, convert to small letter, remove stop words.
clean_result <- dataFrame(path, cleanoutput, 2)
print(clean_result[[1]])
clean_dat <- read.csv(cleanoutput)
clean_df <- popRows(clean_dat)
nrow(clean_df)
clean_df <- clean_df[-c(1:2),]
clean_df <- data.frame(clean_df)
clean_df <- japply(clean_df, which(sapply(clean_df, class)=="factor"), as.character)
clean_df <- japply(clean_df, which(sapply(clean_df, class)=="character"), as.integer)
set.seed(4060)
clean_s <- sample(1:nrow(clean_df),round(0.7*nrow(clean_df)))
training <- clean_df[clean_s,]
testing <- clean_df[-clean_s,]
training$folder
tx <- clean_df[,-which(colnames(training) == "folderTypeClass")]
ty <- clean_df[,which(colnames(training) == "folderTypeClass")]
testx <- clean_df[,-which(colnames(testing) == "folderTypeClass")]
testy <- clean_df[,which(colnames(testing) == "folderTypeClass")]

# apply random forest for feature selection
set.seed(4060)
library(randomForest)
ty <- as.factor(ty)
rf.out <- randomForest(tx,ty, ntree = 10)
rf.pred <- predict(rf.out,newdata = testx)
tb.rf <- table(testy, rf.pred)
# Accuracy of Random Forest after data cleaning
sum(diag(tb.rf))/sum(tb.rf)
# Importance plot
varImpPlot(rf.out, main = "Variable Importance Plot")

# Create dataframe with top 100 most important features according to RF 
set.seed(4060)
library(caret)
v <- varImp(rf.out)
wordsinimp <- rownames(v)
new_v <- cbind(v, wordsinimp) 
top100Imp <- new_v[order(-new_v$Overall),]
top100Imp <- top100Imp[1:100,]
rownames(top100Imp) <- c()
words_in_bow <- c()
for(i in 1:length(top100Imp$wordsinimp)){
  words_in_bow[i] <- which(colnames(clean_df) == top100Imp$wordsinimp[i])
}

words_in_bow <- sort(words_in_bow)
df_imp_features <- clean_df[,words_in_bow]
folderTypeClass <- clean_df$folderTypeClass
df_imp_features <- cbind(df_imp_features,folderTypeClass)
# Divide the new Data Frame into trainig and testing set
set.seed(4060)
sample_index <- sample(1:nrow(df_imp_features), round(0.7*nrow(df_imp_features)))

new_training <- df_imp_features[sample_index,]
new_testing <- df_imp_features[-sample_index,]

classColumn <- which(colnames(df_imp_features) == "folderTypeClass")

# Hyper Parameter tuning using MLR
library(mlr)
set.seed(4060)
rdesc <- makeResampleDesc("CV", iters = 10)
task <- makeClassifTask(data = new_training, target = "folderTypeClass")
ctrl <- makeTuneControlGrid()
set.seed(4060)
# Decision Trees
dt.lrn <- makeLearner("classif.rpart")
dt.ps <- makeParamSet(makeIntegerParam("minsplit", lower = 1, upper = 100),
                      makeIntegerParam("maxdepth", lower = 2, upper = 50),
                      makeDiscreteParam("cp", values = seq(0.001,0.006,0.002)))
dt.res <- tuneParams(dt.lrn, task = task, par.set = dt.ps, control = ctrl, resampling = rdesc, measures = acc)
# Random Forest
set.seed(4060)
rf.lrn <- makeLearner("classif.randomForest")
rf.ps <- makeParamSet(makeIntegerParam("ntree", lower = 100, upper = 200),
                      makeDiscreteParam("mtry", values = c(10,50)))
rf.res <- tuneParams(rf.lrn, task = task, par.set = rf.ps, control = ctrl, resampling = rdesc, measures = acc)

# KNN
set.seed(4060)
knn.lrn <- makeLearner("classif.knn")
knn.ps <- makeParamSet(makeIntegerParam("k", lower = 1, upper = 20))
knn.res <- tuneParams(knn.lrn, task = task, par.set = knn.ps, control = ctrl, resampling = rdesc, measures = acc)

# SVM
set.seed(4060)
svm.lrn <- makeLearner("classif.ksvm")
svm.ps <- makeParamSet(makeDiscreteParam("C", values = c(0.5,1.0,1.5,2.0)),
                       makeDiscreteParam("sigma", values = c(0.5,1.0,1.5,2.0)))
svm.res <- tuneParams(svm.lrn, task = task, par.set = svm.ps, control = ctrl, resampling = rdesc, measures = acc)

# HOLD - OUT Validation
# Naive Bayes
set.seed(4060)
y.hat.index <- 1
y.hat.clean <- c()
words.clean <- colnames(df_imp_features)
for(f in 1:nrow(new_testing)){
  y.hat.clean[y.hat.index] <- naiveBayes(new_testing[f,], words.clean, new_testing)
  y.hat.index <- y.hat.index + 1
}
tb.nb_clean <- table(new_testing[,which(colnames(new_testing) == "folderTypeClass")],y.hat.clean)
# Accuracy
sum(diag(tb.nb_clean))/sum(tb.nb_clean)

# Precision, Recall and F score
class_sum_nb <- c()
class_rsum_nb <- c()
for(i in 1:4){
  class_sum_nb[i] <- tb.nb_clean[i,i]/sum(tb.nb_clean[,i])
}
nb.prec <- 1/(sum(tb.nb_clean)*sum(class_sum_nb))

for(i in 1:4){
  class_rsum_nb[i] <- tb.nb_clean[i,i]/sum(tb.nb_clean[i,])
}
nb.recall <- 1/sum(tb.nb_clean)*sum(class_rsum_nb)
fscore_nb <- 2 * ((nb.recall * nb.prec)/(nb.recall + nb.prec))


# KNN
set.seed(4060)
library(class)
knn.out <- knn(train = new_training[,-classColumn], test = new_testing[,-classColumn], cl = new_training$folderTypeClass, k = knn.res$x$k)
tb <- table(new_testing$folderTypeClass, knn.out)
# Accuracy of KNN
sum(diag(tb))/sum(tb)

# Precision, Recall and F score
class_sum_knn <- c()
class_rsum_knn <- c()
for(i in 1:4){
  class_sum_knn[i] <- tb[i,i]/sum(tb[,i])
}
knn.prec <- 1/(sum(tb)*sum(class_sum_knn))

for(i in 1:4){
  class_rsum_knn[i] <- tb[i,i]/sum(tb[i,])
}
knn.recall <- 1/sum(tb)*sum(class_rsum_knn)
fscore_knn <- 2 * ((knn.recall * knn.prec)/(knn.recall + knn.prec))

plot(knn.out, main = "KNN Bar Plot")

# Random Forest:
set.seed(4060)
rf.out <- randomForest(new_training[,-classColumn], as.factor(new_training$folderTypeClass), ntree = rf.res$x$ntree, mtry = rf.res$x$mtry)
rf.pred <- predict(rf.out, newdata = new_testing[,-classColumn])
rf.tb <- table(new_testing$folderTypeClass, rf.pred)
# Accuracy 
sum(diag(rf.tb))/sum(rf.tb)
# Precision, Recall and F score
class_sum_rf <- c()
class_rsum_rf <- c()
for(i in 1:4){
  class_sum_rf[i] <- rf.tb[i,i]/sum(rf.tb[,i])
}
rf.prec <- 1/(sum(rf.tb)*sum(class_sum_rf))

for(i in 1:4){
  class_rsum_rf[i] <- rf.tb[i,i]/sum(rf.tb[i,])
}
rf.recall <- 1/sum(rf.tb)*sum(class_rsum_rf)
fscore_rf <- 2 * ((rf.recall * rf.prec)/(rf.recall + rf.prec))

plot(rf.out, main = "Random Forest Plot of Error vs Trees")

# Decision Tree
set.seed(4060)
library(rpart)
dt.control <- rpart.control(minsplit = dt.res$x$minsplit, maxdepth = dt.res$x$maxdepth, cp = dt.res$x$cp)
dt.out <- rpart(as.factor(folderTypeClass) ~ ., data = new_training, control = dt.control)
dt.pred <- predict(dt.out, newdata = new_testing[,-classColumn], type = "class")
dt.tb <- table(new_testing$folderTypeClass, dt.pred)
# Accuracy
sum(diag(dt.tb))/sum(dt.tb)
# Precision, Recall and F score
class_sum_dt <- c()
class_rsum_dt <- c()
for(i in 1:4){
  class_sum_dt[i] <- dt.tb[i,i]/sum(dt.tb[,i])
}
dt.prec <- 1/(sum(dt.tb)*sum(class_sum_dt))

for(i in 1:4){
  class_rsum_dt[i] <- dt.tb[i,i]/sum(dt.tb[i,])
}
dt.recall <- 1/sum(dt.tb)*sum(class_rsum_dt)
fscore_dt <- 2 * ((dt.recall * dt.prec)/(dt.recall + dt.prec))

# SVM
set.seed(4060)
library(e1071)
svm.out <- svm(as.factor(new_training$folderTypeClass)~., data = new_training, sigma = svm.res$x$sigma, C = svm.res$x$C)
svm.pred <- predict(svm.out, newdata = new_testing[,-classColumn])
svm.tb <- table(new_testing$folderTypeClass, svm.pred)
# Accuracy
sum(diag(svm.tb))/sum(svm.tb)
# Precision, Recall and F score
class_sum_svm <- c()
class_rsum_svm <- c()
for(i in 1:4){
  class_sum_svm[i] <- svm.tb[i,i]/sum(svm.tb[,i])
}
svm.prec <- 1/(sum(svm.tb)*sum(class_sum_svm))

for(i in 1:4){
  class_rsum_svm[i] <- svm.tb[i,i]/sum(svm.tb[i,])
}
svm.recall <- 1/sum(svm.tb)*sum(class_rsum_svm)
fscore_svm <- 2 * ((svm.recall * svm.prec)/(svm.recall + svm.prec))

# schuffle DF before CV
set.seed(4060)
sample_index <- sample(1:nrow(df_imp_features), nrow(df_imp_features))
df_imp_features <- df_imp_features[sample_index,]
# Cross Validation
folds <- cut(1:nrow(df_imp_features), 10, labels = FALSE)

knn.acc <- c()
rf.acc <- c()
svm.acc <- c()
dt.acc <- c()

for(i in 1:10){
  f <- which(folds == i)
  cv.train <- df_imp_features[-f,]
  cv.test <- df_imp_features[f,]
  knn.cv.out <- knn(cv.train[,-classColumn], cv.test[,-classColumn], cv.train$folderTypeClass, k = knn.res$x$k)
  knn.cv.tb <- table(cv.test$folderTypeClass, knn.cv.out)
  knn.acc[i] <- sum(diag(knn.cv.tb))/sum(knn.cv.tb)
  rf.cv.out <- randomForest(cv.train[,-classColumn], as.factor(cv.train$folderTypeClass), ntree = rf.res$x$ntree, mtry = rf.res$x$mtry)
  rf.cv.pred <- predict(rf.cv.out, newdata = cv.test[,-classColumn])
  rf.cv.tb <- table(cv.test$folderTypeClass, rf.cv.pred)
  rf.acc[i] <- sum(diag(rf.cv.tb))/sum(rf.cv.tb)
  svm.cv.out <- svm(cv.train[,-classColumn], as.factor(cv.train$folderTypeClass), sigma = svm.res$x$sigma, C = svm.res$x$C)
  svm.cv.pred <- predict(svm.cv.out, newdata = cv.test[,-classColumn])
  svm.cv.tb <- table(cv.test$folderTypeClass, svm.cv.pred)
  svm.acc[i] <- sum(diag(svm.cv.tb))/sum(svm.cv.tb)
  dt.cv.out <- rpart(as.factor(folderTypeClass) ~., data = cv.train, control = dt.control)
  dt.cv.pred <- predict(dt.cv.out, newdata= cv.test, type = "class")
  dt.cv.tb <- table(cv.test$folderTypeClass, dt.cv.pred)
  dt.acc[i] <- sum(diag(dt.cv.tb))/sum(dt.cv.tb)
}
plot(knn.acc, type= "b",main = "KNN Accuracy Plot in CV")
mean(knn.acc)
plot(rf.acc, type = "b", main = "Random Forest Plot in CV")
mean(rf.acc)
plot(svm.acc, type = "b", main = "SVM Plot in CV")
mean(svm.acc)
plot(dt.acc, type = "b", main = "Decision Tree Plot in CV")
mean(dt.acc)

# Naive Bayes CV
set.seed(4060)
accuracy.nb.cv = numeric(10)
tb.nb.cv.index <- 1
words <- colnames(df_imp_features)
for(i in 1:10){
  ib = which(folds == i)
  cv.test <- df_imp_features[ib,]
  cv.train <- df_imp_features[-ib,]
  y.hat.index.cv <- 1
  y.hat.cv <- c()
  for(f in 1:nrow(cv.test)){
    y.hat.cv[y.hat.index.cv] <- naiveBayes(cv.test[f,], words, cv.train)
    y.hat.index.cv = y.hat.index.cv + 1
  }
  tb.nb1.cv <- table(cv.test[,which(colnames(cv.test) == "folderTypeClass")], y.hat.cv)
  accuracy.nb.cv[tb.nb.cv.index] <- sum(diag(tb.nb1.cv))/sum(tb.nb1.cv)
  tb.nb.cv.index = tb.nb.cv.index + 1
}
mean(accuracy.nb.cv)

plot(accuracy.nb.cv, type = "b", main = "Naive Bayes Plot in CV")
