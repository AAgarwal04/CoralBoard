# Clears memory
rm(list = ls())
# Clears console
cat("\014")

library(tidyverse)
library(caret)
library(parallel)
library(doParallel)
library(ranger)
library(ggplot2)
library(dplyr)
library(readxl)
library(reshape2)
library(rsample)
library(ggeasy)

cl = makePSOCKcluster(detectCores() - 1) # detectCores() will detect the number of cores from your computer. convention to leave 1 core for OS
registerDoParallel(cl)

dataOG <- read_excel("C:/Users/AgAr082/Documents/Coral/CoralBoard-main/CoralBoard-main/Data/Data.xlsx")
View(dataOG)
data <- select(dataOG, -Inside, -Temp, -UV, -Pressure)
outcome = "Location"
head(data)
set.seed(100)

split = initial_split(data, prop = 0.7, strata = outcome)
data.train  = training(split)
data.test   = testing(split)

# Convert Location to a factor
data.train$Location <- factor(data.train$Location)

# Create the tuning grid for ranger
tuneGrid = expand.grid(
  mtry = 1:9,
  splitrule = "gini",
  min.node.size = 6:9
)

trControl = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  classProbs = TRUE,  # Important for binary classification
  summaryFunction = defaultSummary
)

rfcv = train(Location ~ ., 
             data = data.train,
             method = "ranger",
             trControl = trControl,
             tuneGrid = tuneGrid,
             metric = "Accuracy")

rfcvTable = rfcv$results
maxVal = rfcvTable %>% filter(Accuracy == max(Accuracy))
print(maxVal)
plot(rfcv)

stopCluster(cl) # We need to close the cluster once we are done
registerDoSEQ()

# Get variable importance using ranger package
rf_model <- ranger(Location ~ ., data = data.train, importance = "impurity")
imp <- data.frame(importance = rf_model$variable.importance, 
                  variable = names(data.train)[1:8])
imp <- imp %>% arrange(desc(importance))

# Plot variable importance with optimal mtry and min.node.size values
ggplot(imp) + 
  geom_col(aes(x = reorder(variable,importance), y = importance), fill = "#0096d6") + 
  labs(title = "Variable Importance with Optimal mtry and min.node.size Values",
       x = "Inputs",
       y = "Relative Importance") +
  ggeasy::easy_center_title() + 
  coord_flip()

# Print optimal mtry and min.node.size values
print(paste("Optimal mtry:", maxVal$mtry))
print(paste("Optimal min.node.size:", maxVal$min.node.size))