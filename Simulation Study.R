library(tidyverse)
library(MASS)

sdp <- function(x) {sqrt(mean((x - mean(x)) ^ 2))}
compute_metrics <- function(actual, predicted, levels = 0:5) {
  actual <- factor(actual, levels = levels)
  predicted <- factor(predicted, levels = levels)
  cm <- table(predicted, actual)
  
  accuracy <- sum(diag(cm)) / sum(cm)
  precision <- recall <- f1 <- numeric(length(levels))
  
  wt <- c()
  for (i in seq_along(levels)) {
    TP <- cm[i, i]
    FP <- sum(cm[i, ]) - TP
    FN <- sum(cm[, i]) - TP
    precision[i] <- ifelse(TP + FP == 0, 0, TP / (TP + FP))
    recall[i]    <- ifelse(TP + FN == 0, 0, TP / (TP + FN))
    f1[i] <- ifelse(precision[i] + recall[i] == 0, 0, 
                    2 * precision[i] * recall[i] / (precision[i] + recall[i]))
    wt <- append(wt, length(actual[actual == i - 1]))
  }
  
  macro_precision <- weighted.mean(precision, wt / length(actual))
  macro_recall <- weighted.mean(recall, wt / length(actual))
  macro_f1 <- weighted.mean(f1, wt / length(actual))
  
  return(list(
    accuracy = round(accuracy * 100, 2),
    f1 = round(macro_f1 * 100, 2)
  ))
}

n_year <- 5   #5, 10, 100
n_dpy <- 100   #number of days in each year block
n <- n_year * n_dpy

##### CONSISTENCY #####
### K = 3 ###
set.seed(124)
# model specification
theta0 <- -0.84
theta1 <- 0.84
beta_sin_1 <- 0   #season
beta_cos_1 <- 2     #season
beta_sin_2 <- 0  #diwali
beta_cos_2 <- 0.7   #diwali
beta_sin_3 <- 0 #daytype
beta_cos_3 <- 0.2   #daytype
lag_coef <- 1.9
t_vec <- c(1:n)
sin_1 <- sin(2 * pi * t_vec / 100)  #season
cos_1 <- cos(2 * pi * t_vec / 100)  #season
sin_2 <- sin(4 * pi * t_vec / 100)  #diwali
cos_2 <- cos(4 * pi * t_vec / 100)  #diwali
sin_3 <- sin(2 * pi * t_vec / 5)    #daytype
cos_3 <- cos(2 * pi * t_vec / 5)    #daytype
lc <- (beta_sin_1 * sin_1) + (beta_cos_1 * cos_1) + (beta_sin_2 * sin_2) + 
  (beta_cos_2 * cos_2) + (beta_sin_3 * sin_3) + (beta_cos_3 * cos_3)

# model fitting
theta0_vec <- c()
theta1_vec <- c()
sin_1_vec <- c()
cos_1_vec <- c()
sin_2_vec <- c()
cos_2_vec <- c()
sin_3_vec <- c()
cos_3_vec <- c()
lag_coef_vec <- c()
for (r in c(1:1000)) {
  ## DATA SIMULATION
  y_vec <- c()
  lag_vec <- c()
  y_current <- 1
  for (t in c(1:n)) {
    F0 <- plogis(theta0 - lc[t] - (lag_coef * y_current))
    F1 <- plogis(theta1 - lc[t] - (lag_coef * y_current))
    p0 <- F0
    p1 <- F1 - F0
    p2 <- 1 - p0 - p1
    y <- sample(c(0, 1, 2), size = 1, prob = c(p0, p1, p2))
    lag_vec <- append(lag_vec, y_current)
    y_current <- y
    y_vec <- append(y_vec, y)
  }
  sim_data <- data.frame(y_vec = factor(y_vec, levels = c(0, 1, 2), ordered = T), 
                         sin_1, cos_1, sin_2, cos_2, sin_3, cos_3, 
                         lag_vec)
  
  ## MODEL FITTING
  log_model <- polr(y_vec ~ sin_1 + cos_1 + sin_2 + cos_2 + sin_3 + cos_3 + lag_vec, 
                    data = sim_data, Hess = TRUE)
  
  ## VALUE COLLECTION
  theta0_vec <- append(theta0_vec, as.numeric(log_model$zeta[1]))
  theta1_vec <- append(theta1_vec, as.numeric(log_model$zeta[2]))
  sin_1_vec <- append(sin_1_vec, as.numeric(log_model$coefficients[1]))
  cos_1_vec <- append(cos_1_vec, as.numeric(log_model$coefficients[2]))
  sin_2_vec <- append(sin_2_vec, as.numeric(log_model$coefficients[3]))
  cos_2_vec <- append(cos_2_vec, as.numeric(log_model$coefficients[4]))
  sin_3_vec <- append(sin_3_vec, as.numeric(log_model$coefficients[5]))
  cos_3_vec <- append(cos_3_vec, as.numeric(log_model$coefficients[6]))
  lag_coef_vec <- append(lag_coef_vec, as.numeric(log_model$coefficients[7]))
}

# performance checking
round(c(mean(theta0_vec), sdp(theta0_vec)), 2)
round(c(mean(theta1_vec), sdp(theta1_vec)), 2)
round(c(mean(sin_1_vec), sdp(sin_1_vec)), 2)
round(c(mean(cos_1_vec), sdp(cos_1_vec)), 2)
round(c(mean(sin_2_vec), sdp(sin_2_vec)), 2)
round(c(mean(cos_2_vec), sdp(cos_2_vec)), 2)
round(c(mean(sin_3_vec), sdp(sin_3_vec)), 2)
round(c(mean(cos_3_vec), sdp(cos_3_vec)), 2)
round(c(mean(lag_coef_vec), sdp(lag_coef_vec)), 2)



### K = 4 ###
set.seed(123)
# model specification
theta0 <- -1.1
theta1 <- 0
theta2 <- 1.1
beta_sin_1 <- 0.6   #season
beta_cos_1 <- -1.1     #season
beta_sin_2 <- 1.4  #diwali
beta_cos_2 <- -0.8   #diwali
beta_sin_3 <- -0.1 #daytype
beta_cos_3 <- -0.4   #daytype
lag_coef <- -0.3
t_vec <- c(1:n)
sin_1 <- sin(2 * pi * t_vec / 100)  #season
cos_1 <- cos(2 * pi * t_vec / 100)  #season
sin_2 <- sin(4 * pi * t_vec / 100)  #diwali
cos_2 <- cos(4 * pi * t_vec / 100)  #diwali
sin_3 <- sin(2 * pi * t_vec / 5)    #daytype
cos_3 <- cos(2 * pi * t_vec / 5)    #daytype
lc <- (beta_sin_1 * sin_1) + (beta_cos_1 * cos_1) + (beta_sin_2 * sin_2) + 
  (beta_cos_2 * cos_2) + (beta_sin_3 * sin_3) + (beta_cos_3 * cos_3)

# model fitting
theta0_vec <- c()
theta1_vec <- c()
theta2_vec <- c()
sin_1_vec <- c()
cos_1_vec <- c()
sin_2_vec <- c()
cos_2_vec <- c()
sin_3_vec <- c()
cos_3_vec <- c()
lag_coef_vec <- c()
for (r in c(1:1000)) {
  ## DATA SIMULATION
  y_vec <- c()
  lag_vec <- c()
  y_current <- 2
  for (t in c(1:n)) {
    F0 <- plogis(theta0 - lc[t] - (lag_coef * y_current))
    F1 <- plogis(theta1 - lc[t] - (lag_coef * y_current))
    F2 <- plogis(theta2 - lc[t] - (lag_coef * y_current))
    p0 <- F0
    p1 <- F1 - F0
    p2 <- F2 - F1
    p3 <- 1 - p0 - p1 - p2
    y <- sample(c(0, 1, 2, 3), size = 1, prob = c(p0, p1, p2, p3))
    lag_vec <- append(lag_vec, y_current)
    y_current <- y
    y_vec <- append(y_vec, y)
  }
  sim_data <- data.frame(y_vec = factor(y_vec, levels = c(0, 1, 2, 3), ordered = T), 
                         sin_1, cos_1, sin_2, cos_2, sin_3, cos_3, lag_vec)
  
  ## MODEL FITTING
  log_model <- polr(y_vec ~ sin_1 + cos_1 + sin_2 + cos_2 + sin_3 + cos_3 + lag_vec, 
                    data = sim_data, Hess = TRUE)
  
  ## VALUE COLLECTION
  theta0_vec <- append(theta0_vec, as.numeric(log_model$zeta[1]))
  theta1_vec <- append(theta1_vec, as.numeric(log_model$zeta[2]))
  theta2_vec <- append(theta1_vec, as.numeric(log_model$zeta[3]))
  sin_1_vec <- append(sin_1_vec, as.numeric(log_model$coefficients[1]))
  cos_1_vec <- append(cos_1_vec, as.numeric(log_model$coefficients[2]))
  sin_2_vec <- append(sin_2_vec, as.numeric(log_model$coefficients[3]))
  cos_2_vec <- append(cos_2_vec, as.numeric(log_model$coefficients[4]))
  sin_3_vec <- append(sin_3_vec, as.numeric(log_model$coefficients[5]))
  cos_3_vec <- append(cos_3_vec, as.numeric(log_model$coefficients[6]))
  lag_coef_vec <- append(lag_coef_vec, as.numeric(log_model$coefficients[7]))
}

# performance checking
round(c(mean(theta0_vec), sdp(theta0_vec)), 2)
round(c(mean(theta1_vec), sdp(theta1_vec)), 2)
round(c(mean(theta2_vec), sdp(theta2_vec)), 2)
round(c(mean(sin_1_vec), sdp(sin_1_vec)), 2)
round(c(mean(cos_1_vec), sdp(cos_1_vec)), 2)
round(c(mean(sin_2_vec), sdp(sin_2_vec)), 2)
round(c(mean(cos_2_vec), sdp(cos_2_vec)), 2)
round(c(mean(sin_3_vec), sdp(sin_3_vec)), 2)
round(c(mean(cos_3_vec), sdp(cos_3_vec)), 2)
round(c(mean(lag_coef_vec), sdp(lag_coef_vec)), 2)


##### FORECASTING #####
### K = 3 ###
set.seed(123)
# model specification
theta0 <- -0.8
theta1 <- 0.78
beta_season1 <- 0.2
beta_season2 <- -1.4
beta_seasonality2 <- 0.3
beta_seasonality3 <- 0.9
lag_coef <- 1.1
t_vec <- c(1:n)
Is.Season1 <- rep(replace(rep(0, 100), 1:30, 1), n_year)  #Season 1
Is.Season2 <- rep(replace(rep(0, 100), 31:70, 1), n_year)   #Season 2
Is.Seasonality2 <- rep(rep(replace(rep(0, 5), 4:5, 1), 20), n_year)   #Weekend
Is.Seasonality3 <- rep(replace(rep(0, 100), 81:85, 1), n_year)    #Diwali
lc <- (beta_season1 * Is.Season1) + (beta_season2 * Is.Season2) + 
  (beta_seasonality2 * Is.Seasonality2) + (beta_seasonality3 * Is.Seasonality3)

# model fitting
accuracy_vec_1 <- c()
f1_vec_1 <- c()
accuracy_vec_2 <- c()
f1_vec_2 <- c()
for (r in c(1:1000)) {
  ## DATA SIMULATION
  y_vec <- c()
  lag_vec <- c()
  y_current <- 1
  for (t in c(1:n)) {
    F0 <- plogis(theta0 - lc[t] - (lag_coef * y_current))
    F1 <- plogis(theta1 - lc[t] - (lag_coef * y_current))
    p0 <- F0
    p1 <- F1 - F0
    p2 <- 1 - p0 - p1
    y <- sample(c(0, 1, 2), size = 1, prob = c(p0, p1, p2))
    lag_vec <- append(lag_vec, y_current)
    y_current <- y
    y_vec <- append(y_vec, y)
  }
  model_data <- data.frame(t = t_vec, Response = factor(y_vec, levels = c(0, 1, 2), ordered = T), 
                           Is.Season1, Is.Season2, Is.Seasonality2, Is.Seasonality3, Lag1 = lag_vec) %>% 
    mutate(Seasonality1_sin = sin(2 * pi * t / 100), Seasonality1_cos = cos(2 * pi * t / 100), 
           Seasonality2_sin = sin(2 * pi * t / 5), Seasonality2_cos = cos(2 * pi * t / 5), 
           Seasonality3_sin = abs(sin(1 * pi * t / 100)), Seasonality3_cos = abs(cos(1 * pi * t / 100)))
  
  ## TRAIN-TEST SPLITTING
  test_size <- n * 20 / 100
  train_data <- model_data[1:(n - test_size), ]
  test_data <- model_data[(n - (test_size - 1)):n, ]
  
  ## MODEL FITTING OVER TRAIN DATA
  log_model_1 <- polr(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + Is.Seasonality3 + Lag1, 
                      data = train_data, Hess = TRUE)
  log_model_2 <- polr(Response ~ Seasonality1_sin + Seasonality1_cos + Seasonality2_sin + Seasonality2_cos + 
                      Seasonality3_sin + Seasonality3_cos + Lag1, data = train_data, Hess = TRUE)
  
  ## FORECASTING OVER TEST DATA
  actual_log <- as.numeric(test_data$Response) - 1
  pred_1_log <- as.numeric(predict(log_model_1, test_data)) - 1
  pred_2_log <- as.numeric(predict(log_model_2, test_data)) - 1
  
  ## VALUE COLLECTION
  metrics_1 <- compute_metrics(actual_log, pred_1_log)
  accuracy_vec_1 <- append(accuracy_vec_1, metrics_1$accuracy)
  f1_vec_1 <- append(f1_vec_1, metrics_1$f1)
  metrics_2 <- compute_metrics(actual_log, pred_2_log)
  accuracy_vec_2 <- append(accuracy_vec_2, metrics_2$accuracy)
  f1_vec_2 <- append(f1_vec_2, metrics_2$f1)
}

# performance checking
round(c(mean(accuracy_vec_1), mean(f1_vec_1)), 2)
round(c(mean(accuracy_vec_2), mean(f1_vec_2)), 2)



### K = 4 ###
set.seed(126)
# model specification
theta0 <- -0.9
theta1 <- 0.02
theta2 <- 1.2
beta_season1 <- -0.4
beta_season2 <- -1.3
beta_seasonality2 <- 0.5
beta_seasonality3 <- 0.7
lag_coef <- 1.2
t_vec <- c(1:n)
Is.Season1 <- rep(replace(rep(0, 100), 1:30, 1), n_year)  #Season 1
Is.Season2 <- rep(replace(rep(0, 100), 31:70, 1), n_year)   #Season 2
Is.Seasonality2 <- rep(rep(replace(rep(0, 5), 4:5, 1), 20), n_year)   #Weekend
Is.Seasonality3 <- rep(replace(rep(0, 100), 81:85, 1), n_year)    #Diwali
lc <- (beta_season1 * Is.Season1) + (beta_season2 * Is.Season2) + 
  (beta_seasonality2 * Is.Seasonality2) + (beta_seasonality3 * Is.Seasonality3)

# model fitting
accuracy_vec_1 <- c()
f1_vec_1 <- c()
accuracy_vec_2 <- c()
f1_vec_2 <- c()
for (r in c(1:1000)) {
  ## DATA SIMULATION
  y_vec <- c()
  lag_vec <- c()
  y_current <- 2
  for (t in c(1:n)) {
    F0 <- plogis(theta0 - lc[t] - (lag_coef * y_current))
    F1 <- plogis(theta1 - lc[t] - (lag_coef * y_current))
    F2 <- plogis(theta2 - lc[t] - (lag_coef * y_current))
    p0 <- F0
    p1 <- F1 - F0
    p2 <- F2 - F1
    p3 <- 1 - p0 - p1 - p2
    y <- sample(c(0, 1, 2, 3), size = 1, prob = c(p0, p1, p2, p3))
    lag_vec <- append(lag_vec, y_current)
    y_current <- y
    y_vec <- append(y_vec, y)
  }
  model_data <- data.frame(t = t_vec, Response = factor(y_vec, levels = c(0, 1, 2, 3), ordered = T), 
                           Is.Season1, Is.Season2, Is.Seasonality2, Is.Seasonality3, Lag1 = lag_vec) %>% 
    mutate(Seasonality1_sin = sin(2 * pi * t / 100), Seasonality1_cos = cos(2 * pi * t / 100), 
           Seasonality2_sin = sin(2 * pi * t / 5), Seasonality2_cos = cos(2 * pi * t / 5), 
           Seasonality3_sin = abs(sin(1 * pi * t / 100)), Seasonality3_cos = abs(cos(1 * pi * t / 100)))
  
  ## TRAIN-TEST SPLITTING
  test_size <- n * 20 / 100
  train_data <- model_data[1:(n - test_size), ]
  test_data <- model_data[(n - (test_size - 1)):n, ]
  
  ## MODEL FITTING OVER TRAIN DATA
  log_model_1 <- polr(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + Is.Seasonality3 + Lag1, 
                      data = train_data, Hess = TRUE)
  log_model_2 <- polr(Response ~ Seasonality1_sin + Seasonality1_cos + Seasonality2_sin + Seasonality2_cos + 
                      Seasonality3_sin + Seasonality3_cos + Lag1, data = train_data, Hess = TRUE)
  
  ## FORECASTING OVER TEST DATA
  actual_log <- as.numeric(test_data$Response) - 1
  pred_1_log <- as.numeric(predict(log_model_1, test_data)) - 1
  pred_2_log <- as.numeric(predict(log_model_2, test_data)) - 1
  
  ## VALUE COLLECTION
  metrics_1 <- compute_metrics(actual_log, pred_1_log)
  accuracy_vec_1 <- append(accuracy_vec_1, metrics_1$accuracy)
  f1_vec_1 <- append(f1_vec_1, metrics_1$f1)
  metrics_2 <- compute_metrics(actual_log, pred_2_log)
  accuracy_vec_2 <- append(accuracy_vec_2, metrics_2$accuracy)
  f1_vec_2 <- append(f1_vec_2, metrics_2$f1)
}

# performance checking
round(c(mean(accuracy_vec_1), mean(f1_vec_1)), 2)
round(c(mean(accuracy_vec_2), mean(f1_vec_2)), 2)
