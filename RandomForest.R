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
  geom_col(aes(x = reorder(variable,importance), y = importance), fill = "#549EF8") + 
  labs(title = "Variable Importance",
       x = "Inputs",
       y = "Relative Importance") +
  ggeasy::easy_center_title() + 
  coord_flip() +
  theme(
    axis.text = element_text(size = 22),       # Increase axis value size
    axis.text.y = element_text(size = 22),     # Adjust y-axis text (variable names) separately if needed
    axis.title = element_text(size = 24),      # Increase axis title size
    plot.title = element_text(size = 26),      # Increase plot title size
    plot.margin = margin(10, 10, 10, 10, "pt") # Add some margin around the plot
  ) +
  theme(aspect.ratio = 0.9)  # Adjust this value to change the chart's aspect ratio
# Print optimal mtry and min.node.size values
print(paste("Optimal mtry:", maxVal$mtry))
print(paste("Optimal min.node.size:", maxVal$min.node.size))