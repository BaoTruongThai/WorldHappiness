
# 1.Data Loading and Exploration
## 1.1 Data Loading
library(ggplot2)
library(glmnet)
library(tidyverse)
library(corrplot)
library(dplyr)
library(pROC)
library(reshape2)
library(caret)


df <- read.csv('happiness.csv')
dim(df)

## 1.2 Data Exploration
### 1.2.1 Dataset Overview
head(df)
str(df)
summary(df)


### 1.2.2 Top 10 countries with highest average Life Ladder

avg.ll <- aggregate(df$Life.Ladder ~ 
                      df$Country.name, 
                    data=df, 
                    FUN=mean)

# Removes "df$" from the column names
names(avg.ll) <- gsub("df\\$", "", names(avg.ll))



top10 <- head(avg.ll[order(-avg.ll$Life.Ladder), ], 10)

# Subset original data to just those countries
hap_top10 <- merge(df, top10, by = "Country.name")

# Boxplot for top 10
boxplot(Life.Ladder.x ~ Country.name, data = hap_top10,
        las = 2,
        cex.axis = 0.7,
        col = "red",
        main = "Average Life Ladder in Top 10 Countries",
        xlab = "Country",
        ylab = "Life Ladder Score") 


### 1.2.3 Average Life Ladder per Year
# average Life ladder per Year
avg.year <- aggregate(df$Life.Ladder ~ 
                        df$Year, 
                      data=df, 
                      FUN=mean)

library(dplyr)
c1 <- count(df,Year)

plot(c1[,1], avg.year[,2],
     xlab="Year",
     ylab = "Average Life Ladder", 
     main='Average Life Ladder per Year',
     type = 'b',
     pch=19,
     col='purple'
     )

### 1.2.4 Freedom vs Life Ladder
plot(Life.Ladder~Freedom.to.make.life.choices, 
     data = df,
     main='Freedom vs Life Ladder')

FvLL.m <- lm(Life.Ladder ~ Freedom.to.make.life.choices, data = df)

abline(FvLL.m, col = "red", lwd = 2)


# 2. Data Processing 

## 2.1 Remove Irrelevant Features
### 2.1.1 Drop Country Name and Year column
columns_to_drop <- c(
  'Country.name',
  'Year'
)

df <- df %>% select(-all_of(columns_to_drop))

df

colnames(df)

### 2.1.2 Check null values percentage
na_percentage <- function(df) {
  missing_percentage <- list()
  total <- nrow(df)
  
  for (col in names(df)) {
    missing_value <- sum(is.na(df[[col]]))
    percentage <- (missing_value * 100) / total
    if (percentage > 0) {
      missing_percentage[[col]] <- percentage
    }
  }
  
  return(missing_percentage)
}

# Usage (equivalent to your Python code)
na_list <- na_percentage(df)
print(na_list)

### 2.1.3 Drop columns with high null values percentage

columns_to_drop <- c(
  'Standard.deviation.of.ladder.by.country.year',
  'Standard.deviation.Mean.of.ladder.by.country.year',
  'GINI.index..World.Bank.estimate.',
  'GINI.index..World.Bank.estimate...average.2000.16',
  'gini.of.household.income.reported.in.Gallup..by.wp5.year',
  'Most.people.can.be.trusted..Gallup',
  'Most.people.can.be.trusted..WVS.round.1981.1984',
  'Most.people.can.be.trusted..WVS.round.1989.1993',
  'Most.people.can.be.trusted..WVS.round.1994.1998',
  'Most.people.can.be.trusted..WVS.round.1999.2004',
  'Most.people.can.be.trusted..WVS.round.2005.2009',
  'Most.people.can.be.trusted..WVS.round.2010.2014'
)

# Drop the columns
df_cleaned <- df %>% select(-all_of(columns_to_drop))

df_cleaned

num_col <- df_cleaned %>% select(where(is.numeric))
corr_matrix <- cor(num_col, use = "complete.obs")

corr_melted <- melt(corr_matrix)

p <- ggplot(corr_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 2)), 
            size = 15, 
            color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 30, color = "white"),  # X-axis labels
    axis.text.y = element_text(size = 30, color = "white"),                        # Y-axis labels
    plot.title = element_text(hjust = 0.5, size = 20),           # Title
    legend.title = element_text(size = 30, color = "white"),                      # Legend title
    legend.text = element_text(size = 20, color = "white")) +
  labs(title = "Correlation Matrix", x = "", y = "") +
  coord_fixed()


ggsave("correlation_matrix.png", plot = p, width = 30, height = 30, dpi = 300)

print(p)

# As we can observe the correlation between Confidence in national government and Life Ladder is quite low so we will consider to drop this feature
## Drop
columns_to_drop <- c(
  'Confidence.in.national.government'
)

# Drop the columns
df_cleaned <- df_cleaned %>% select(-all_of(columns_to_drop))

df_cleaned

## 2.2 Handle Missing data
### 2.2.1 Check missing data percentage

na_list <- na_percentage(df_cleaned)
print(na_list)
n_before <- nrow(df_cleaned)
df_cleaned <- df_cleaned %>% drop_na()
n_after  <- nrow(df_cleaned)
message("Dropped ", n_before - n_after, " rows (", round((n_before-n_after)/n_before*100,1), "%).")

## 2.3 Duplicate data

df %>%
  group_by_all() %>%
  filter(n()>1) %>%
  ungroup()


detect_outliers_iqr <- function(df, multiplier = 1.5) {
  # Get numeric columns only
  numeric_cols <- sapply(df, is.numeric)
  df_numeric <- df[, numeric_cols, drop = FALSE]
  
  # Create summary dataframe
  results <- data.frame(
    Column = names(df_numeric),
    Outliers = NA,
    Percentage = NA,
    stringsAsFactors = FALSE
  )
  
  for (i in 1:ncol(df_numeric)) {
    col_name <- names(df_numeric)[i]
    x <- df_numeric[[i]]
    x <- x[!is.na(x)]  # Remove NAs
    
    if (length(x) > 0) {
      # Calculate IQR bounds
      Q1 <- quantile(x, 0.25)
      Q3 <- quantile(x, 0.75)
      IQR <- Q3 - Q1
      lower <- Q1 - multiplier * IQR
      upper <- Q3 + multiplier * IQR
      
      # Count outliers
      outliers <- sum(x < lower | x > upper)
      percentage <- round((outliers / length(x)) * 100, 1)
      
      results$Outliers[i] <- outliers
      results$Percentage[i] <- percentage
    }
  }
  
  return(results)
}

outliers <- detect_outliers_iqr(df_cleaned)

outliers
df_cleaned$Happiness.Level <- cut(df_cleaned$Life.Ladder,
                                     breaks = c(2, 4, 6, 9),
                                     labels = c("Low Happiness", "Medium Happiness", "High Happiness"),
                                     right = FALSE)

df_cleaned
table(df_cleaned$Happiness.Level)
prop.table(table(df_cleaned$Happiness.Level))

create_corruption_boxplot <- function(df) {
  ggplot(df, aes(y = Perceptions.of.corruption)) +
    geom_boxplot(fill = "lightcoral", alpha = 0.7, outlier.color = "red", outlier.size = 2) +
    labs(
      title = "Boxplot: Perceptions of Corruption",
      y = "Perceptions of Corruption",
      x = ""
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
      axis.text.y = element_text(size = 12),
      axis.title.y = element_text(size = 14)
    )
}

create_corruption_boxplot(df_cleaned)

set.seed(123)
### First split
# 90% train_val, 10% test_labeled
sample_size <- floor(0.9 * nrow(df_cleaned))

train_val_indices <- sample(nrow(df_cleaned), sample_size)

# Split the data
train_val_data <- df_cleaned[train_val_indices, ]
test_labeled_data <- df_cleaned[-train_val_indices, ]

### Second split
# 80% train, 20% val
sample_size <- floor(0.8 * nrow(train_val_data))

train_indices <- sample(nrow(train_val_data), sample_size)

# Split the data into train and val dataset
train_data <- train_val_data[train_indices, ]
val_data <- train_val_data[-train_indices, ]


## 3.2 Cross Validation
### 3.2.1 Regression Model

library(glmnet)
set.seed(123)
## Assign X_train and y_train
X_train.regression <- model.matrix(Life.Ladder ~ Log.GDP.per.capita + Social.support + Healthy.life.expectancy.at.birth + Freedom.to.make.life.choices + Generosity + Perceptions.of.corruption + Positive.affect + Negative.affect + Democratic.Quality + Delivery.Quality, data=train_data)[, -1]
y_train.regression <- train_data$Life.Ladder


## Assign X_val and y_val
X_val.regression <- model.matrix(Life.Ladder ~ Log.GDP.per.capita + Social.support + Healthy.life.expectancy.at.birth + Freedom.to.make.life.choices + Generosity + Perceptions.of.corruption + Positive.affect + Negative.affect + Democratic.Quality + Delivery.Quality, data=val_data)[, -1]
y_val.regression <- val_data$Life.Ladder



#### 3.2.1.1 Scale

preproc <- preProcess(X_train.regression, method = c("center", "scale"))

X_train.regression <- predict(preproc, X_train.regression)

X_val.regression <- predict(preproc, X_val.regression)


#### 3.2.1.2 Ridge

# Calculate cross-validation to find lambda
cv_ridge.regression <- cv.glmnet(X_train.regression, y_train.regression, alpha = 0, nfolds = 5)

plot(cv_ridge.regression, ylab="Mean-Squared Error", main = "Ridge Regression Cross-Validation")

# Get optimal lambda
lambda_ridge_min.regression <- cv_ridge.regression$lambda.min

cat("Ridge Regression Results:\n")
cat("Lambda min (lowest CV error):", lambda_ridge_min.regression, "\n")

#### Fit model with optimal lambda for Ridge

ridge_model.regression <- glmnet(X_train.regression, y_train.regression, alpha = 0, lambda = lambda_ridge_min.regression)

ridge_coefs.regression <- coef(ridge_model.regression)
cat("\nRidge Regression Coefficients (lambda.min):\n")
print(ridge_coefs.regression)


#### 3.2.1.3 Lasso

# Calculate cross-validation to find lambda
cv_lasso.regression <- cv.glmnet(X_train.regression, y_train.regression, alpha = 1, nfolds = 5)

plot(cv_lasso.regression, main = "Lasso Regression Cross-Validation")

# Get optimal lambda
lambda_lasso_min.regression <- cv_lasso.regression$lambda.min

cat("Lambda min (lowest CV error):", lambda_lasso_min.regression, "\n")


#### Fit model with optimal lambda for Lasso

lasso_model.regression <- glmnet(X_train.regression, y_train.regression, alpha = 1, lambda = lambda_lasso_min.regression)

lasso_coefs.regression <- coef(lasso_model.regression)
cat("\nLasso Regression Coefficients (lambda.min):\n")
print(lasso_coefs.regression)



### 3.2.2 Classification Model

library(class)     # for knn()
library(caret)     

## Assign X_train and y_train
set.seed(123)
X_train.classification <- model.matrix(Happiness.Level ~ Log.GDP.per.capita + Social.support + Healthy.life.expectancy.at.birth + Freedom.to.make.life.choices + Generosity + Perceptions.of.corruption + Positive.affect + Negative.affect + Democratic.Quality + Delivery.Quality, data=train_data)[, -1]
y_train.classification <- train_data$Happiness.Level

## Assign X_val and y_val
X_val.classification <- model.matrix(Happiness.Level ~ Log.GDP.per.capita + Social.support + Healthy.life.expectancy.at.birth + Freedom.to.make.life.choices + Generosity + Perceptions.of.corruption + Positive.affect + Negative.affect + Democratic.Quality + Delivery.Quality, data=val_data)[, -1]
y_val.classification <- val_data$Happiness.Level


# Scale features (important for KNN)
preproc <- preProcess(X_train.classification, method = c("center", "scale"))
X_train.classification <- predict(preproc, X_train.classification)
X_val.classification <- predict(preproc, X_val.classification)



#### 3.2.2.1 Ridge
# Calculate cross-validation to find lambda
cv_ridge.classification <- cv.glmnet(X_train.classification, y_train.classification, alpha = 0, family = "multinomial")

## Ridge
plot(cv_ridge.classification, ylab="Mean-Squared Error", main = "Ridge Classification Cross-Validation")

# Get optimal lambda
lambda_ridge_min.classification <- cv_ridge.classification$lambda.min

cat("Ridge Classification Results:\n")
cat("Lambda min (lowest CV error):", lambda_ridge_min.classification, "\n")


#### Fit model with optimal lambda

ridge_model.classification <- glmnet(X_train.classification, y_train.classification, alpha = 0, lambda = lambda_ridge_min.classification, family = "multinomial")

ridge_coefs.classification <- coef(ridge_model.classification)
cat("\nRidge Classification Coefficients (lambda.min):\n")
print(ridge_coefs.classification)

#### 3.2.2.2 Lasso

# Calculate cross-validation to find lambda
cv_lasso.classification <- cv.glmnet(X_train.classification, y_train.classification, alpha = 1, family = "multinomial")

# Lasso
plot(cv_lasso.classification, main = "Lasso Classification Cross-Validation")

# Get optimal lambda
lambda_lasso_min.classification <- cv_lasso.classification$lambda.min

cat("Lambda min (lowest CV error):", lambda_lasso_min.classification, "\n")

#### Fit model with optimal lambda

lasso_model.classification <- glmnet(X_train.classification, y_train.classification, alpha = 1, lambda = lambda_lasso_min.classification, family = "multinomial")

lasso_coefs.classification <- coef(lasso_model.classification)
cat("\nLasso Classification Coefficients (lambda.min):\n")
print(lasso_coefs.classification)

## 3.3 Training model
### 3.3.1 Regression model
#### 3.3.1.1 Linear Regression

X_train_with_intercept <- cbind(Intercept = 1, X_train.regression)
X_val_with_intercept   <- cbind(Intercept = 1, X_val.regression)

# Step 2: Fit linear regression using low-level lm.fit()
lm_model <- lm.fit(x = X_train_with_intercept, y = y_train.regression)

# Step 3: Predict on validation set
y_pred <- as.vector(X_val_with_intercept %*% lm_model$coefficients)


# Calculate MSE, RMSE, R-squared
lm_mse <- mean((y_pred - y_val.regression)^2)
lm_rmse <- sqrt(lm_mse)
lm_r2 <- cor(y_val.regression, y_pred)^2
cat("Linear Regression: \n")
cat("MSE: ", lm_mse, "\nRMSE: ", lm_rmse, "\nR-squared: ", lm_r2, "\n")


#### 3.3.1.2 RandomForest Regression

library(randomForest)
# Fit the model
set.seed(123)

rf_model <- randomForest(x = X_train.regression,
                         y = y_train.regression,
                         ntree = 500,          
                         mtry = 3,             
                         importance = TRUE)

# Predict
rf_pred <- predict(rf_model, newdata = X_val.regression)

# Calculate MSE, RMSE, R-squared
rf_mse <- mean((rf_pred - y_val.regression)^2)
rf_rmse <- sqrt(rf_mse)
rf_r2 <- cor(y_val.regression, rf_pred)^2
cat("Random Forest:", "\n")

cat("MSE: ", rf_mse, "\nRMSE: ", rf_rmse, "\nR-squared: ", rf_r2, "\n")

# View variable importance
importance(rf_model)
varImpPlot(rf_model)


### 3.3.2 Classification model - KNN

set.seed(123)
knn_pred <- knn(train = X_train.classification,
                test = X_val.classification,
                cl = y_train.classification,
                k = 5)


# Evaluate
conf_mat <- confusionMatrix(knn_pred, y_val.classification)
print(conf_mat)


table(knn_pred)
table(y_val.classification)




## 3.4 Training model with test_labeled dataset

### 3.4.1 Regression model

## Assign X_train_val and y_train_val
X_train_val.regression <- model.matrix(Life.Ladder ~ Log.GDP.per.capita + Social.support + Healthy.life.expectancy.at.birth + Freedom.to.make.life.choices + Generosity + Perceptions.of.corruption + Positive.affect + Negative.affect + Democratic.Quality + Delivery.Quality, data=train_val_data)[, -1]
y_train_val.regression <- train_val_data$Life.Ladder

## Assign X_test_labeled and y_test_labeled
X_test_labeled.regression <- model.matrix(Life.Ladder ~ Log.GDP.per.capita + Social.support + Healthy.life.expectancy.at.birth + Freedom.to.make.life.choices + Generosity + Perceptions.of.corruption + Positive.affect + Negative.affect + Democratic.Quality + Delivery.Quality, data=test_labeled_data)[, -1]
y_test_labeled.regression <- test_labeled_data$Life.Ladder

### Scale data
X_train_val.regression <- predict(preproc, X_train_val.regression)
X_test_labeled.regression <- predict(preproc, X_test_labeled.regression)


#### 3.4.1.1 Linear Regression

X_train_val_w_intercept <- cbind(Intercept = 1, X_train_val.regression)
X_test_labeled_w_intercept   <- cbind(Intercept = 1, X_test_labeled.regression)

# Step 2: Fit linear regression using low-level lm.fit()
lm_model <- lm.fit(x = X_train_val_w_intercept, y = y_train_val.regression)

# Step 3: Predict on validation set
final_y_pred <- as.vector(X_test_labeled_w_intercept %*% lm_model$coefficients)


# Calculate MSE, RMSE, R-squared
final_lm_mse <- mean((final_y_pred - y_test_labeled.regression)^2)
final_lm_rmse <- sqrt(final_lm_mse)
final_lm_r2 <- cor(y_test_labeled.regression, final_y_pred)^2
cat("Linear Regression: \n")
cat("MSE: ", final_lm_mse, "\nRMSE: ", final_lm_rmse, "\nR-squared: ", final_lm_r2, "\n")


#### 3.4.1.2 Random Forest

# Fit the model
set.seed(123)

final_rf_model <- randomForest(x = X_train_val.regression,
                         y = y_train_val.regression,
                         ntree = 500,          
                         mtry = 3,             
                         importance = TRUE)

# Predict
final_rf_pred <- predict(final_rf_model, newdata = X_test_labeled.regression)

# Calculate MSE, RMSE, R-squared
final_rf_mse <- mean((final_rf_pred - y_test_labeled.regression)^2)
final_rf_rmse <- sqrt(final_rf_mse)
final_rf_r2 <- cor(y_test_labeled.regression, final_rf_pred)^2
cat("Random Forest:", "\n")

cat("MSE: ", final_rf_mse, "\nRMSE: ", final_rf_rmse, "\nR-squared: ", final_rf_r2, "\n")

# View variable importance
importance(final_rf_model)
varImpPlot(final_rf_model)

### 3.4.2 Classification

## Assign X_train_val and y_train_val
X_train_val.classification <- model.matrix(Happiness.Level ~ Log.GDP.per.capita + Social.support + Healthy.life.expectancy.at.birth + Freedom.to.make.life.choices + Generosity + Perceptions.of.corruption + Positive.affect + Negative.affect + Democratic.Quality + Delivery.Quality, data=train_val_data)[, -1]
y_train_val.classification <- train_val_data$Happiness.Level


## Assign X_test_labeled and y_test_labeled
X_test_labeled.classification <- model.matrix(Happiness.Level ~ Log.GDP.per.capita + Social.support + Healthy.life.expectancy.at.birth + Freedom.to.make.life.choices + Generosity + Perceptions.of.corruption + Positive.affect + Negative.affect + Democratic.Quality + Delivery.Quality, data=test_labeled_data)[, -1]
y_test_labeled.classification <- test_labeled_data$Happiness.Level


# Scaling
X_train_val.classification <- predict(preproc, X_train_val.classification)
X_test_labeled.classification <- predict(preproc, X_test_labeled.classification)


set.seed(123)
result_knn_pred <- knn(train = X_train_val.classification,
                test = X_test_labeled.classification,
                cl = y_train_val.classification,
                k = 5)

# Evaluate
result_conf_mat.classification <- confusionMatrix(result_knn_pred, y_test_labeled.classification)
print(result_conf_mat.classification)

