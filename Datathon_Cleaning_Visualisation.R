library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(zoo)
library(dplyr)
library(readr)

# Load the dataset and convert it to a tibble
data <- read_csv("Quarterly Data.csv", col_types = cols()) %>% as_tibble()

# Remove the first two metadata rows
data <- data[-c(1, 2), ]

# Ensure 'sasdate' (or the actual date column name) remains unchanged
date_col <- "sasdate"  # Update this if your column name is different

# Convert only numeric columns, excluding 'sasdate'
data <- data %>%
  mutate(across(setdiff(names(data), date_col), ~ suppressWarnings(as.numeric(.))))

# Start from the 115th row (after adjusting for removed metadata rows)
data <- data[113:nrow(data), ]

# Remove columns with missing values (but keep 'sasdate')
data <- data %>% select(sasdate, where(~ all(!is.na(.))))

# Create the GDP binary indicator column
data <- data %>% mutate(GDP_Binary = as.integer(GDPC1 > lag(GDPC1)))

# Print first few rows to check if 'sasdate' is preserved
print(head(data))




# Step 1: Remove Variables Through Methods (Example: Remove Variables with Over 50% Missing Values)
threshold <- 0.5  # 50% missing value threshold
data_cleaned <- data %>% select(where(~ mean(is.na(.)) < threshold))

# Step 2: Compute Correlation Matrix (Only Numeric Variables)
numeric_data <- data_cleaned %>% select(where(is.numeric))
cor_matrix <- cor(numeric_data, use = "pairwise.complete.obs")

# Step 3: Remove Highly Correlated Variables
high_cor_vars <- findCorrelation(cor_matrix, cutoff = 0.9, verbose = FALSE)
data_cleaned <- data_cleaned %>% select(-all_of(names(numeric_data)[high_cor_vars]))

# Step 4: Check for Low Variance Using Standard Deviation
low_variance_vars <- numeric_data %>% summarise(across(everything(), sd, na.rm = TRUE))
low_variance_vars <- names(low_variance_vars)[which(low_variance_vars < 0.01)]  # Adjust threshold as needed
data_cleaned <- data_cleaned %>% select(-all_of(low_variance_vars))

# Step 5: Perform PCA
# Fix missing and infinite values
data_cleaned <- data_cleaned %>%
  mutate(across(everything(), ~ ifelse(is.infinite(.), NA, .))) %>%
  drop_na() %>%
  select(where(is.numeric))  # Keep only numeric variables

# Save the 'sasdate' column separately
sasdate_col <- data$sasdate[113:nrow(data)]  # Assuming the date column should align with your cleaned data

# Run PCA
pca_model <- prcomp(data_cleaned, scale. = TRUE, center = TRUE)

# Check PCA summary
summary(pca_model)

# Step 6: Streamline Variables Using PCA Loadings
loadings <- as.data.frame(pca_model$rotation)  # PCA loadings
top_features <- rownames(loadings)[apply(abs(loadings), 1, max) > 0.2]  # Keep features with high influence
data_final <- data_cleaned %>% select(all_of(top_features))

# Ensure that we extract the last 150 rows of 'sasdate'
sasdate_col <- data$sasdate[(nrow(data) - 149):nrow(data)]  # Last 150 rows of sasdate

# Add 'sasdate' back to the cleaned data
data_final <- bind_cols(sasdate = sasdate_col, data_cleaned)

# Display Results
print("Selected Variables after PCA:")
print(top_features)

# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(ROCR)  # Required for performance-based ROC plotting

# Select only the specified features + GDP_Binary
selected_features <- c("A014RE1Q156NBEA", "A823RL1Q225SBEA", "IPNMAT", "IPNCONGD",
                       "CUMFNS", "USGOOD", "USCONS", "USINFO", "USMINE", "USWTRADE", "GDP_Binary")

data_filtered <- data_final %>% select(all_of(selected_features))

# Ensure GDP_Binary is correctly formatted: 0 = Recession, 1 = No Recession
data_filtered <- data_filtered %>%
  mutate(Recession = as.factor(GDP_Binary)) %>%
  select(-GDP_Binary)  # Remove GDP_Binary after creating Recession column

# Explicitly set factor levels to ensure correct ROC interpretation
data_filtered$Recession <- factor(data_filtered$Recession, levels = c("0", "1"))

# Split the data into training (80%) and testing (20%)
set.seed(123)
train_index <- createDataPartition(data_filtered$Recession, p = 0.8, list = FALSE)
train_data <- data_filtered[train_index, ]
test_data <- data_filtered[-train_index, ]

# Train a Logistic Regression model using only the selected features
glm_model <- glm(Recession ~ ., data = train_data, family = binomial)

# Get probability predictions
glm1_pred <- predict(glm_model, test_data, type = "response")  # Probabilities

# Compute ROC Curve
pred <- prediction(glm1_pred, test_data$Recession)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")  # TPR vs. FPR
auc_perf <- performance(pred, measure = "auc")  # Calculate AUC

# Plot ROC curve
plot(perf, col = "steelblue", lwd = 2, main = "ROC Curve for Recession Prediction")
abline(0, 1, lwd = 1, lty = 2, col = "gray")  # Add dashed diagonal line

# Display AUC on the plot
auc_value <- round(auc_perf@y.values[[1]], 2)
text(0.4, 0.8, paste("AUC =", auc_value), col = "black", cex = 1.2)



#DATA VISUALISATION AND ANALYSIS
# Define the selected features
selected_features <- c("A014RE1Q156NBEA", "A823RL1Q225SBEA", "IPNMAT", "IPNCONGD", "CUMFNS",
                       "USGOOD", "USCONS", "USINFO", "USMINE", "USWTRADE")

# Step 1, 2, and 3 combined: Create lagged GDP binary, clean the data, and reshape it to long format
data_lagged_long <- data %>%
  mutate(GDP_Binary_Lagged = lag(GDP_Binary, 3)) %>%  # Lag the GDP binary by 3 months
  filter(!is.na(GDP_Binary_Lagged)) %>%  # Remove rows with missing GDP_Binary_Lagged
  drop_na() %>%  # Drop rows with any NA in remaining columns
  gather(key = "Feature", value = "Value", -sasdate, -GDP_Binary_Lagged) %>%
  filter(Feature %in% selected_features)  # Filter to include only the selected features

# Ensure sasdate is converted to Date format for plotting purposes only
data_lagged_long <- data_lagged_long %>%
  mutate(sasdate = as.Date(sasdate, format = "%m/%d/%Y"))

# Plot the data for the top 10 selected features, lagged GDP binary, and recession indicator
# Add filled bars for recession periods
library(ggplot2)
library(dplyr)

ggplot(data_lagged_long, aes(x = sasdate, y = Value, color = Feature, group = Feature)) +
  geom_line(size = 1.2) + 
  geom_line(aes(y = GDP_Binary_Lagged), color = "blue", linetype = "solid", size = 1.5) +  # Ensure blue line is solid and on top
  geom_rect(data = data_lagged_long %>% filter(GDP_Binary_Lagged == 0), 
            aes(xmin = sasdate, xmax = sasdate + 60, ymin = -Inf, ymax = Inf), 
            fill = "pink", alpha = 0.15, inherit.aes = FALSE) +  # Adjust transparency
  labs(title = "Top 10 Selected Features and Lagged GDP Binary (3 Months)", x = "Date", y = "Value") +
  theme_minimal() +
  theme(legend.position = "right")  # Keep legend readable


# Subset the data to include selected features and GDP_Binary_Lagged
selected_features <- c("A014RE1Q156NBEA", "A823RL1Q225SBEA", "IPNMAT", "IPNCONGD", "CUMFNS",
                       "USGOOD", "USCONS", "USINFO", "USMINE", "USWTRADE")

# Ensure to add GDP_Binary_Lagged to the selected features
features_with_gdp <- c(selected_features, "GDP_Binary_Lagged")

# Subset the data
data_subset <- data_lagged %>%
  select(all_of(features_with_gdp))

# Compute the correlation matrix for the selected features and GDP_Binary_Lagged
cor_matrix <- cor(data_subset, use = "complete.obs")

# Reshape the correlation matrix into long format for ggplot
corr_melted <- melt(cor_matrix)

# Plot the correlation heatmap
ggplot(corr_melted, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Correlation Heatmap of Selected Features with Lagged GDP Binary", x = "Features", y = "Features") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

library(tidyverse)
library(lubridate)

# Step 1: Load the FCTAX annual data
fctax_annual <- read_csv("FCTAX (1).csv")

# Convert 'observation_date' to Date type
fctax_annual <- fctax_annual %>%
  mutate(observation_date = as.Date(observation_date)) %>%
  mutate(year = year(observation_date))

# Drop NA values and ensure 'year' is an integer
fctax_annual <- fctax_annual %>%
  filter(!is.na(year)) %>%
  mutate(year = as.integer(year))

# Step 2: Convert annual FCTAX to quarterly data
quarters_data <- tibble()
for (i in 1:nrow(fctax_annual)) {
  yr <- fctax_annual$year[i]
  val <- fctax_annual$FCTAX[i]
  
  # Define the start of each quarter
  q_starts <- c(
    as.Date(paste(yr, "01", "01", sep = "-")),
    as.Date(paste(yr, "04", "01", sep = "-")),
    as.Date(paste(yr, "07", "01", sep = "-")),
    as.Date(paste(yr, "10", "01", sep = "-"))
  )
  
  # Append rows to quarters_data
  quarters_data <- bind_rows(quarters_data, tibble(sasdate = q_starts, FCTAX = rep(val, 4)))
}

# Step 3: Shift each quarter by 2 months
quarters_data <- quarters_data %>%
  mutate(sasdate = sasdate %m+% months(2))

# Step 4: Drop rows before 1959-03-01 and sort by date
quarters_data <- quarters_data %>%
  filter(sasdate >= as.Date("1959-03-01")) %>%
  arrange(sasdate)

# Step 5: Load your main data (df)
df <- data_lagged_long

# Ensure the 'sasdate' column is in Date format
df <- df %>%
  mutate(sasdate = as.Date(sasdate))

# Merge the FCTAX quarterly data with the main DataFrame
df_merged <- left_join(df, quarters_data, by = "sasdate")

# Step 6: Plot FCTAX vs GDP Binary Indicator over time
ggplot(df_merged, aes(x = sasdate)) +
  # Plot FCTAX as a line
  geom_line(aes(y = FCTAX, color = "FCTAX"), size = 1) +
  # Plot points for GDP Growth (when GDP_Binary_Lagged is 1)
  geom_point(data = df_merged %>% filter(GDP_Binary_Lagged == 1), aes(y = FCTAX, color = "GDP Growth"), size = 1) +
  # Plot points for GDP Contraction (when GDP_Binary_Lagged is 0)
  geom_point(data = df_merged %>% filter(GDP_Binary_Lagged == 0), aes(y = FCTAX, color = "GDP Contraction"), size = 1) +
  # Customize the plot
  scale_color_manual(values = c("FCTAX" = "blue", "GDP Growth" = "green", "GDP Contraction" = "red")) +
  labs(title = "FCTAX vs GDP Growth Indicator Over Time",
       x = "Year",
       y = "FCTAX",
       color = "Legend") +
  theme_minimal() +
  theme(legend.position = "top")

