# OLS
# find the ground truth
# Load the dataset
train_data <- read.csv("train_house.csv")
# Construct OLS regression model
ols_model <- lm(medv ~ . - ID, data = train_data)
# Display regression results (including significance tests)
summary(ols_model)



# Lasso Regression model
library(glmnet)  
library(dplyr)   
data <- read.csv("housing_data.csv")
predictors <- c("crim", "indus", "age", "zn", "chas", "nox", "rm", "dis", 
                "rad", "tax", "ptratio", "lstat")
response <- "medv"
vars_to_standardize <- c("crim", "indus", "age", "zn", "chas", "nox", "rm", "dis", 
                         "rad", "tax", "ptratio", "lstat", "medv")
data[vars_to_standardize] <- scale(data[vars_to_standardize])
true_relevant_features <- c("zn", "chas", "nox", "rm", "dis", "rad", "tax", "ptratio", "lstat")
irrelevant_features <- c("crim", "indus", "age")
type1_errors <- numeric(100)
type2_errors <- numeric(100)
selected_variables_all <- c()  
set.seed(123)
for (i in 1:100) {
  sample_data <- data[sample(nrow(data), 180, replace = FALSE), ]
  X <- as.matrix(sample_data[, predictors])
  y <- sample_data[, response]
  lasso_model <- cv.glmnet(X, y, alpha = 1, family = "gaussian")
  selected_features <- predictors[which(coef(lasso_model, s = "lambda.min")[-1] != 0)]
  selected_variables_all <- c(selected_variables_all, selected_features)
  false_positives <- setdiff(selected_features, true_relevant_features)
  type1_errors[i] <- length(false_positives) / length(selected_features)
  false_negatives <- setdiff(true_relevant_features, selected_features)
  type2_errors[i] <- length(false_negatives) / length(true_relevant_features)
}
mean_type1_error <- mean(type1_errors, na.rm = TRUE)
mean_type2_error <- mean(type2_errors, na.rm = TRUE)
variable_frequency <- sort(table(selected_variables_all), decreasing = TRUE)
cat("Average Type 1 error rate:", mean_type1_error, "\n")
cat("Average Type 2 error rate:", mean_type2_error, "\n")
cat("Frequencies of All Selected Variables:\n")
print(variable_frequency)


# Load necessary libraries
library(caret)  # For stratified sampling
library(glmnet)  # For Lasso and Adaptive Lasso

# Load the dataset
data <- read.csv("housing_data.csv")

# Define true variables
true_variables <- c("zn", "chas", "nox", "rm", "dis", "rad", "tax", "ptratio", "lstat")

# Initialize storage for results
set.seed(123)
n_simulations <- 100  # Number of simulations
fdr_values <- numeric(n_simulations)  # Store FDR for each simulation
type2_error_values <- numeric(n_simulations)  # Store Type II Error for each simulation
all_variables <- colnames(data)[!(colnames(data) %in% c("ID", "medv"))]
selection_counts <- data.frame(Variable = all_variables, Selected_Count = rep(0, length(all_variables)))  # Initialize counts

# Simulation loop
for (i in 1:n_simulations) {
  # Step 1: Stratified sampling of 180 data points
  data$medv_bins <- cut(data$medv, breaks = quantile(data$medv, probs = seq(0, 1, 0.1)), include.lowest = TRUE)
  sample_index <- createDataPartition(data$medv_bins, p = 180 / nrow(data), list = FALSE)
  sampled_data <- data[sample_index, ]
  sampled_data$medv_bins <- NULL  # Remove bin column
  
  # Step 2: Prepare predictors and response
  X <- as.matrix(sampled_data[, !(names(sampled_data) %in% c("ID", "medv"))])
  y <- sampled_data$medv
  
  # Step 3: Initial Lasso with 5-fold cross-validation to compute weights
  initial_model <- cv.glmnet(X, y, alpha = 1, standardize = TRUE, nfolds = 5)
  initial_coefs <- abs(coef(initial_model, s = "lambda.min")[-1])  # Exclude intercept
  weights <- 1 / (initial_coefs + 1e-6)  # Add small constant to avoid division by zero
  
  # Step 4: Fit Adaptive Lasso with 5-fold cross-validation
  adaptive_model <- cv.glmnet(X, y, alpha = 1, penalty.factor = weights, standardize = TRUE, nfolds = 5)
  adaptive_coefs <- coef(adaptive_model, s = "lambda.min")
  
  # Step 5: Filter variables with non-zero coefficients
  coef_df <- data.frame(
    Variable = rownames(adaptive_coefs),
    Coefficient = as.numeric(adaptive_coefs)
  )
  coef_df <- coef_df[coef_df$Variable != "(Intercept)", ]  # Exclude intercept
  non_zero_coefs <- coef_df[coef_df$Coefficient != 0, ]
  selected_variables <- non_zero_coefs$Variable
  
  # Step 6: Calculate FDR and Type II Error
  false_positives <- setdiff(selected_variables, true_variables)
  false_negatives <- setdiff(true_variables, selected_variables)
  total_selected <- length(selected_variables)
  total_true <- length(true_variables)
  
  fdr <- ifelse(total_selected > 0, length(false_positives) / total_selected, 0)
  type2_error <- length(false_negatives) / total_true
  
  # Record FDR and Type II Error for this simulation
  fdr_values[i] <- fdr
  type2_error_values[i] <- type2_error
  
  # Update selection counts for variables
  selection_counts$Selected_Count <- selection_counts$Selected_Count + as.numeric(selection_counts$Variable %in% selected_variables)
}

# Step 7: Compute averages
avg_fdr <- mean(fdr_values)
avg_type2_error <- mean(type2_error_values)

# Print results
print(paste("Average FDR:", avg_fdr))
print(paste("Average Type II Error Rate:", avg_type2_error))
print("FDR Values for Each Simulation:")
print(fdr_values)
print("Type II Error Values for Each Simulation:")
print(type2_error_values)

# Prepare the selection counts table
print("Selection Counts Table:")
print(selection_counts)

# Rank variables based on Selected_Count in descending order
selection_counts_ranked <- selection_counts[order(-selection_counts$Selected_Count), ]

# Print the ranked table
print("Ranked Selection Counts Table:")
print(selection_counts_ranked)

### Print the ranked table with only the first 9 variables
print("Top 9 Variables with the Highest Selection Counts:")
print(head(selection_counts_ranked, 9))



# SCAD model
library(ncvreg)
data <- read.csv("housing_data.csv")
predictors <- c("crim", "indus", "age", "zn", "chas", "nox", "rm", "dis", 
                "rad", "tax", "ptratio", "lstat")
response <- "medv"
vars_to_standardize <- c(predictors, response)
data[vars_to_standardize] <- scale(data[vars_to_standardize])
true_relevant_features <- c("zn", "chas", "nox", "rm", "dis", "rad", "tax", "ptratio", "lstat")
irrelevant_features <- c("crim", "indus", "age")
type1_errors <- numeric(100)
type2_errors <- numeric(100)
selected_variables_all <- c()  
cv_errors <- numeric(100)
set.seed(123)
for (i in 1:100) {
  sample_data <- data[sample(nrow(data), 180, replace = FALSE), ]
  X <- as.matrix(sample_data[, predictors])
  y <- sample_data[, response]
  scad_model <- cv.ncvreg(X, y, penalty = "SCAD", family = "gaussian")
  lambda_avg <- mean(c(scad_model$lambda.min, scad_model$lambda.1se), na.rm = TRUE)
  selected_features <- predictors[which(scad_model$fit$beta[-1, which.min(scad_model$cve)] != 0)]
  selected_variables_all <- c(selected_variables_all, selected_features)
  cv_errors[i] <- min(scad_model$cve)
  false_positives <- setdiff(selected_features, true_relevant_features)
  type1_errors[i] <- ifelse(length(selected_features) > 0, 
                            length(false_positives) / length(selected_features), NA)
  false_negatives <- setdiff(true_relevant_features, selected_features)
  type2_errors[i] <- length(false_negatives) / length(true_relevant_features)
}
mean_type1_error <- mean(type1_errors, na.rm = TRUE)
mean_type2_error <- mean(type2_errors, na.rm = TRUE)
mean_cv_error <- mean(cv_errors, na.rm = TRUE)
variable_frequency <- sort(table(selected_variables_all), decreasing = TRUE)
cat("Average Type 1 error rate:", mean_type1_error, "\n")
cat("Average Type 2 error rate:", mean_type2_error, "\n")
cat("Frequencies of All Selected Variables:\n")