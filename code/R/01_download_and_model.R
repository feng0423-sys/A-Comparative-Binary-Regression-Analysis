# STAT 4052 Final Project - Cervical cancer biopsy screening analysis
# Primary R script for downloading, cleaning, fitting, tuning, and evaluating models.
# The report was generated from the same modeling design: train/test split, median
# imputation with missingness indicators, logistic regression, radial SVM, and random forest.

set.seed(4052)

required_packages <- c(
  "tidyverse", "caret", "pROC", "e1071", "randomForest", "knitr"
)
missing_packages <- setdiff(required_packages, rownames(installed.packages()))
if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}
invisible(lapply(required_packages, library, character.only = TRUE))

project_dir <- normalizePath(file.path(getwd()), mustWork = FALSE)
data_dir <- file.path(project_dir, "data")
output_dir <- file.path(project_dir, "outputs")
figure_dir <- file.path(project_dir, "figures")
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

# Dataset source: UCI Machine Learning Repository, Cervical Cancer (Risk Factors), id 383.
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
data_path <- file.path(data_dir, "risk_factors_cervical_cancer.csv")
if (!file.exists(data_path)) {
  download.file(url, data_path, mode = "wb")
}

raw <- readr::read_csv(data_path, na = "?", show_col_types = FALSE)
dat <- raw %>% mutate(across(everything(), as.numeric))

# Define the response and predictors.
# Biopsy is the binary endpoint. Hinselmann, Schiller, and Citology are excluded
# because they are alternative diagnostic outcomes/tests rather than baseline risk factors.
target <- "Biopsy"
diagnostic_cols <- c("Hinselmann", "Schiller", "Citology")
missing_rate <- sapply(dat, function(x) mean(is.na(x)))
high_missing_cols <- names(missing_rate[missing_rate > 0.50])
predictors <- setdiff(names(dat), c(target, diagnostic_cols, high_missing_cols))

model_df <- dat %>% select(all_of(c(target, predictors)))
model_df <- model_df %>%
  mutate(Biopsy = factor(if_else(Biopsy == 1, "positive", "negative"),
                         levels = c("positive", "negative")))

# Add missingness indicators before median imputation.
cols_with_missing <- predictors[sapply(model_df[predictors], function(x) any(is.na(x)))]
for (nm in cols_with_missing) {
  model_df[[paste0(make.names(nm), "_missing")]] <- as.integer(is.na(model_df[[nm]]))
}

# Stratified split: 70% training, 30% held-out testing.
idx <- caret::createDataPartition(model_df$Biopsy, p = 0.70, list = FALSE)
train_df <- model_df[idx, , drop = FALSE]
test_df <- model_df[-idx, , drop = FALSE]

# Median-impute numeric predictors using training medians only.
numeric_predictors <- setdiff(names(train_df), "Biopsy")
train_medians <- sapply(train_df[numeric_predictors], function(x) median(x, na.rm = TRUE))
for (nm in numeric_predictors) {
  train_df[[nm]][is.na(train_df[[nm]])] <- train_medians[[nm]]
  test_df[[nm]][is.na(test_df[[nm]])] <- train_medians[[nm]]
}

# Remove any zero-variance predictors after imputation.
zv <- caret::nearZeroVar(train_df[numeric_predictors])
if (length(zv) > 0) {
  drop_zv <- numeric_predictors[zv]
  train_df <- train_df %>% select(-all_of(drop_zv))
  test_df <- test_df %>% select(-all_of(drop_zv))
}

# Cross-validation setup. sampling = "up" balances the positive/negative classes
# inside each training fold only, preventing leakage from validation folds.
ctrl <- caret::trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling = "up",
  verboseIter = FALSE
)

# Logistic regression. caret's glm uses no tuning parameters, but CV estimates ROC.
fit_logit <- caret::train(
  Biopsy ~ ., data = train_df,
  method = "glm",
  family = binomial(),
  metric = "ROC",
  trControl = ctrl,
  preProcess = c("center", "scale")
)

# Radial-kernel SVM with a compact tuning grid.
svm_grid <- expand.grid(
  C = c(0.5, 2, 10),
  sigma = c(0.01, 0.03, 0.10)
)
fit_svm <- caret::train(
  Biopsy ~ ., data = train_df,
  method = "svmRadial",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = svm_grid,
  preProcess = c("center", "scale")
)

# Random forest with a compact tuning grid. ntree = 75 matches the fast reproducible
# analysis used to create the report outputs.
rf_grid <- expand.grid(mtry = unique(pmax(1, round(c(sqrt(ncol(train_df) - 1), 0.5 * (ncol(train_df) - 1))))))
fit_rf <- caret::train(
  Biopsy ~ ., data = train_df,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = rf_grid,
  ntree = 75,
  importance = TRUE
)

models <- list(
  "Logistic regression" = fit_logit,
  "Radial SVM" = fit_svm,
  "Random forest" = fit_rf
)

# Select thresholds from cross-validated training predictions only.
filter_best <- function(pred_df, best_tune) {
  out <- pred_df
  if (ncol(best_tune) > 0) {
    for (nm in names(best_tune)) {
      out <- out[out[[nm]] == best_tune[[nm]], , drop = FALSE]
    }
  }
  out
}

youden_threshold <- function(obs, prob_positive) {
  roc_obj <- pROC::roc(response = obs, predictor = prob_positive,
                       levels = c("negative", "positive"), direction = "<", quiet = TRUE)
  best <- pROC::coords(roc_obj, x = "best", best.method = "youden",
                       ret = c("threshold", "sensitivity", "specificity"), transpose = FALSE)
  tibble(threshold = as.numeric(best$threshold),
         sensitivity = as.numeric(best$sensitivity),
         specificity = as.numeric(best$specificity),
         auc = as.numeric(pROC::auc(roc_obj)))
}

evaluate_model <- function(name, fit, threshold) {
  prob <- predict(fit, newdata = test_df, type = "prob")$positive
  pred <- factor(if_else(prob >= threshold, "positive", "negative"),
                 levels = c("positive", "negative"))
  cm <- caret::confusionMatrix(pred, test_df$Biopsy, positive = "positive")
  roc_obj <- pROC::roc(response = test_df$Biopsy, predictor = prob,
                       levels = c("negative", "positive"), direction = "<", quiet = TRUE)
  tibble(
    method = name,
    threshold = threshold,
    auc = as.numeric(pROC::auc(roc_obj)),
    accuracy = unname(cm$overall["Accuracy"]),
    balanced_accuracy = unname(cm$byClass["Balanced Accuracy"]),
    sensitivity = unname(cm$byClass["Sensitivity"]),
    specificity = unname(cm$byClass["Specificity"]),
    precision = unname(cm$byClass["Precision"]),
    f1 = unname(cm$byClass["F1"]),
    tp = cm$table["positive", "positive"],
    fp = cm$table["positive", "negative"],
    fn = cm$table["negative", "positive"],
    tn = cm$table["negative", "negative"]
  )
}

threshold_rows <- list()
metric_rows <- list()
for (nm in names(models)) {
  fit <- models[[nm]]
  pred <- filter_best(fit$pred, fit$bestTune)
  thr <- youden_threshold(pred$obs, pred$positive)
  threshold_rows[[nm]] <- thr %>% mutate(method = nm, .before = 1)
  metric_rows[[paste0(nm, "_default")]] <- evaluate_model(nm, fit, 0.50) %>% mutate(threshold_type = "0.50")
  metric_rows[[paste0(nm, "_youden")]] <- evaluate_model(nm, fit, thr$threshold[[1]]) %>% mutate(threshold_type = "CV Youden")
}

threshold_table <- bind_rows(threshold_rows)
metric_table <- bind_rows(metric_rows)
readr::write_csv(threshold_table, file.path(output_dir, "training_thresholds_R.csv"))
readr::write_csv(metric_table, file.path(output_dir, "test_metrics_R.csv"))

# Basic plots used for diagnostics.
png(file.path(figure_dir, "class_balance_R.png"), width = 1300, height = 850, res = 200)
barplot(table(model_df$Biopsy), main = "Biopsy outcome distribution", ylab = "Number of patients")
dev.off()

roc_list <- lapply(models, function(fit) {
  prob <- predict(fit, newdata = test_df, type = "prob")$positive
  pROC::roc(response = test_df$Biopsy, predictor = prob,
            levels = c("negative", "positive"), direction = "<", quiet = TRUE)
})
png(file.path(figure_dir, "roc_curves_R.png"), width = 1300, height = 1000, res = 200)
plot(roc_list[[1]], col = 1, main = "Test-set ROC curves")
plot(roc_list[[2]], col = 2, add = TRUE)
plot(roc_list[[3]], col = 3, add = TRUE)
legend("bottomright", legend = names(models), col = 1:3, lwd = 2)
dev.off()

saveRDS(models, file.path(output_dir, "fitted_models_R.rds"))
print(metric_table)
