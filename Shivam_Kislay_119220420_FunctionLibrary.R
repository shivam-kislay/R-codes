####
# Install Packages
####
install.packages("class")
install.packages("e1071")
install.packages("randomForest")
install.packages("caret")
install.packages("mlr")
install.packages("stopwords")
install.packages("taRifx")
####
# Libraries
####
library(stopwords)

library(taRifx)
####

# Functions

# to add filter to which words to be included in the data set
wordFilter <- function(word, stpWrds){
  x <- gsub("[[:punct:]]", "", word)
  x <- gsub("\\w*[0-9]+\\w*\\s*", "", x)
  if(x == "") return(x)
  x <- tolower(x)
  check <- which(stpWrds == x)
  if(length(check) != 0 | nchar(x) <= 3) return("")
  else
    return(x)
}


# Function to load the word frequqncy in a dataset
data.load <- function(FilePath, operation){
  count <- c()
  words <- c()
  # files <- list.files(Folderpath)
  index <- 1
  new_index <- 0
  found <- FALSE
  stpwrds <- stopwords(language = "en")
  
  data <- file(FilePath, open = "r")
  lines <- readLines(data)
  
  word_library <- c()
  word_library_index <- 1
  
  # loop to read each line of the file that is read
  for(i in 1:length(lines)){
    if(lines[i] == "" | lines[i] == " ") {
      next
    }
    line = strsplit(lines[i], " ")[[1]]
    
    # loop to read each word in the line
    for(j in line){
      if (j == "" | j == " "){
        next
      }
      if(operation == 2){
        j <- wordFilter(j, stpwrds)
        if(grepl("^\\s*$", j)) next
      }
      word_library[word_library_index] <- j
      word_library_index <- word_library_index + 1
      if(length(words) > 0 & length(count) > 0){
        # loop to skim through the word vector to check if the word is already added in that vector
        for(o in 1:length(words)){
          if(j == words[o]){
            found = TRUE
            new_index = o
            break
          }
        }
        # Update only the index of the count if word already in word vector
        if(found == TRUE){
          count[new_index] = count[new_index] + 1
          found = FALSE
        }
        # else add the word in the word vector and set count to 1 in count vector
        else{
          words[index] = j
          count[index] = 1
          index = index + 1
        }
        # this is when first word is added in the vectors  
      }
      else{
        words[index] = j
        count[index] = 1
        index = index + 1
      }
    }
  }
  # close each file after it is read, else code will give warning!!!
  close(data)
  # }
  # Count vector and word vector should be of the same length else throw error
  if(length(words) != length(count)){
    print("Word and Count Dictionary Not Equal")
  }
  else{
    
    unsorted_obj_data <- data.frame(words, count)
    
    sorted_obj_data <- unsorted_obj_data[order(-count),]
    
    return(list(sorted_obj_data, word_library))
  }
}


# funtion to skim through the files in the folder and return a dataframe with 
# unique words from each file with their class, file path and frequencies
dataFrame <- function(path, outputPath, operation){
  folder <- list.dirs(path)
  folder_index = 1
  d <- data.frame()
  word_library <- c()
  buffer <- c()
  for(f in folder){
    if(f == path) next
    files <- list.files(f)
    for(file in files){
      p <- paste(f,file,sep ="//")
      if(operation == 1){
        result<- data.load(p,1)
        df <- result[[1]]
        buffer <- result[[2]]
      }
      else{
        result <- data.load(p,2)
        df <- result[[1]]
        buffer <- result[[2]]
      }
      filez <- rep(p, nrow(df))
      classifier <- folder_index
      datFrame <- data.frame(df,filez,classifier)
      d <- rbind(d,datFrame)
      word_library <- c(word_library, buffer)
    }
    folder_index = folder_index + 1
  }
  write.csv(d, outputPath)
  memo <- "values loaded"
  return(list(memo,word_library))
}

# function to create the bag of words
popRows <- function(dat){
  vocab <- as.vector(unique(dat$words))
  df <- t(data.frame(table(vocab)))
  colnames(df) <- df[1,]
  rownames(df) <- c()
  level <- levels(as.factor(dat$filez))
  folderTypeClass <- c(rep(0, nrow(df)))
  df <- cbind(df, folderTypeClass)
  classifierIndex <- which(colnames(df) == "folderTypeClass")
  for(l in level){
    set <- which(dat$filez == l)
    row <- rep(0, ncol(df))
    for(i in set){
      index <- which(colnames(df) == dat$words[i])
      row[index] <- dat$count[i]
      row[classifierIndex] <- dat$classifier[i]
    }
    df <- rbind(df, row)
    df[which(colnames(df) == "folderTypeClass")] = 1
  }
  return(df)
}

# Funciton to implement naive bayes manually
naiveBayes <- function(f, words, dat){
  row <- f
  class1 <- dat[which(dat$folderTypeClass == 1),]
  class2 <- dat[which(dat$folderTypeClass == 2),]
  class3 <- dat[which(dat$folderTypeClass == 3),]
  class4 <- dat[which(dat$folderTypeClass == 4),]
  pClass1 <- nrow(class1)/nrow(dat)
  pClass2 <- nrow(class2)/nrow(dat)
  pClass3 <- nrow(class3)/nrow(dat)
  pClass4 <- nrow(class4)/nrow(dat)
  wordsinc1 <- sum(colSums(class1))
  wordsinc2 <- sum(colSums(class2))
  wordsinc3 <- sum(colSums(class3))
  wordsinc4 <- sum(colSums(class4))
  p1 <- c()
  p2 <- c()
  p3 <- c()
  p4 <- c()
  index <- 1
  for(i in 1:length(row)){
    if(row[i] == 0) next
    word <- words[i]
    # inC1 <- which(class1$words == word)
    inC1 <- which(colnames(class1) == word)
    if(length(inC1) == 0){su1 = 0}
    else{
      su1 <- sum(class1[,inC1])
    }
    nb1 <- ((su1 + 1)/(wordsinc1 + length(words)))
    p1[index] <- nb1
    inC2 <- which(colnames(class2) == word)
    if(length(inC2) == 0){su2 = 0}
    else{
      su2 <- sum(class2[,inC2])
    }
    nb2 <- ((su2 + 1)/(wordsinc2 + length(words)))
    p2[index] <- nb2
    inC3 <- which(colnames(class3) == word)
    if(length(inC3) == 0){su3 = 0}
    else{
      su3 <- sum(class3[,inC3])
    }
    nb3 <- ((su3 + 1)/(wordsinc3 + length(words)))
    p3[index] <- nb3
    inC4 <- which(colnames(class4) == word)
    if(length(inC4) == 0){su4 = 0}
    else{
      su4 <- sum(class4[,inC4])
    }
    nb4 <- ((su4 + 1)/(wordsinc4 + length(words)))
    p4[index] <- nb4
    index <- index + 1
  }
  return(which.max(c(pClass1 * prod(p1),pClass2 * prod(p2), pClass3 * prod(p3), pClass4 * prod(p4))))
}

