# !!! Run Methods.R file before running codes of this file !!!

library(microbenchmark)
library(knitr)

R <- 1000

n_year <- 5   #5, 10, 100
n_dpy <- 100   #number of days in each year block
n <- n_year * n_dpy

##### FORECASTING #####
### K = 3 ###
Sys.time()
#set.seed(123)
# MODEL SPECIFICATION
theta0 <- -0.8
theta1 <- 0.78
beta_season1 <- 0.2
beta_season2 <- -1.4
beta_seasonality2 <- 0.3
beta_seasonality3 <- 0.9
gamma00 <- 0.9
gamma01 <- -0.2
gamma10 <- 0.1
gamma11 <- -0.7
t_vec <- c(1:n)
Is.Season1 <- rep(replace(rep(0, 100), 1:30, 1), n_year)  #Season 1
Is.Season2 <- rep(replace(rep(0, 100), 31:70, 1), n_year)   #Season 2
Is.Seasonality2 <- rep(rep(replace(rep(0, 5), 4:5, 1), 20), n_year)   #Weekend
Is.Seasonality3 <- rep(replace(rep(0, 100), 81:85, 1), n_year)    #Diwali
lc <- (beta_season1 * Is.Season1) + (beta_season2 * Is.Season2) + 
  (beta_seasonality2 * Is.Seasonality2) + (beta_seasonality3 * Is.Seasonality3)

acc_tsolr <- c()
f1_tsolr <- c()
acc_isolr <- c()
f1_isolr <- c()
acc_mtd <- c()
f1_mtd <- c()
acc_par <- c()
f1_par <- c()
acc_rf <- c()
f1_rf <- c()
acc_svm <- c()
f1_svm <- c()
time_tsolr <- c()
time_isolr <- c()
time_mtd <- c()
time_par <- c()
time_rf <- c()
time_svm <- c()
for (iter in c(1:R)) {
  
  cat("Running Iteration", iter, "out of", R, "\n")
  
  ## DATA SIMULATION
  y_vec <- rep(0, (n + 1))
  y_vec[1] <- sample(c(0, 1, 2), 1)
  for (t in c(2:(n + 1))) {
    I0 <- as.numeric(y_vec[t - 1] == 0)
    I1 <- as.numeric(y_vec[t - 1] == 1) # CATEGORY 2 IS BASELINE
    eta0 <- lc[t - 1] + (gamma00 * I0) + (gamma01 * I1)
    eta1 <- lc[t - 1] + (gamma10 * I0) + (gamma11 * I1)
    F0 <- plogis(theta0 - eta0)
    F1 <- plogis(theta1 - eta1)
    p0 <- F0
    p1 <- F1 - F0
    p2 <- 1 - p0 - p1
    y_vec[t] <- sample(c(0, 1, 2), size = 1, prob = c(p0, p1, p2))
  }
  lag_y = y_vec[-length(y_vec)]
  
  model_data <- data.frame(t = t_vec, y_vec = y_vec[-1], 
                           Response = factor(y_vec[-1], levels = c(0, 1, 2), ordered = T), 
                           Is.Season1, Is.Season2, Is.Seasonality2, Is.Seasonality3, 
                           lag_y = lag_y) %>% 
    mutate(I0 = as.numeric(lag_y == 0), I1 = as.numeric(lag_y == 1))
  
  # TRAIN-TEST SPLIT
  test_size <- n * 20 / 100
  train_data <- model_data[1:(n - test_size), ]
  test_data <- model_data[(n - (test_size - 1)):n, ]
  
  # TSOLR
  best_k <- tsolr_1(y = train_data$Response, num_seas = 2, frequency = c(5, 100))$best_K
  tsolr_model <- tsolr_2(y = train_data$Response, best_K = best_k, frequency = c(5, 100))
  pred_tsolr <- predict_tsolr(model = tsolr_model, train_y = train_data$Response, best_K = best_k,
                              test_y = test_data$Response, frequency = c(5, 100))$predicted_class
  m_tsolr <- compute_metrics(actual = test_data$Response, predicted = pred_tsolr, levels = 0:2)
  acc_tsolr <- append(acc_tsolr, m_tsolr$accuracy)
  f1_tsolr <- append(f1_tsolr, m_tsolr$f1)
  
  # ISOLR
  isolr_model <- vglm(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + Is.Seasonality3 + I0 + I1,
                      family = cumulative(link = "logitlink", parallel = ~ Is.Season1 + Is.Season2 + 
                                            Is.Seasonality2 + Is.Seasonality3 - 1, reverse = FALSE),
                      data = train_data)
  pred_isolr <- predict(isolr_model, newdata = test_data, type = "response")
  pred_isolr <- as.numeric(colnames(pred_isolr)[max.col(pred_isolr)])
  m_isolr <- compute_metrics(actual = test_data$Response, predicted = pred_isolr, levels = 0:2)
  acc_isolr <- append(acc_isolr, m_isolr$accuracy)
  f1_isolr <- append(f1_isolr, m_isolr$f1)
  
  # MTD
  pred_mtd <- mtd_fn(y = model_data$y_vec, y.training = train_data$y_vec, 
                     y.test = test_data$y_vec)
  m_mtd <- compute_metrics(actual = test_data$y_vec, predicted = pred_mtd, levels = 0:2)
  acc_mtd <- append(acc_mtd, m_mtd$accuracy)
  f1_mtd <- append(f1_mtd, m_mtd$f1)
  
  # PAR
  pred_par <- par_fn(y = model_data$y_vec, y.training = train_data$y_vec, 
                     y.test = test_data$y_vec)
  m_par <- compute_metrics(actual = test_data$y_vec, predicted = pred_par, levels = 0:2)
  acc_par <- append(acc_par, m_par$accuracy)
  f1_par <- append(f1_par, m_par$f1)
  
  # RF
  rf_model <- randomForest(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + 
                             Is.Seasonality3 + lag_y, data = train_data)
  pred_rf <- predict(rf_model, test_data)
  m_rf <- compute_metrics(actual = test_data$Response, predicted = pred_rf, levels = 0:2)
  acc_rf <- append(acc_rf, m_rf$accuracy)
  f1_rf <- append(f1_rf, m_rf$f1)
  
  # SVM
  svm_model <- svm(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + 
                     Is.Seasonality3 + lag_y, data = train_data, type = 'C', kernel = "sigmoid")
  pred_svm <- predict(svm_model, test_data)
  m_svm <- compute_metrics(actual = test_data$Response, predicted = pred_svm, levels = 0:2)
  acc_svm <- append(acc_svm, m_svm$accuracy)
  f1_svm <-append(f1_svm, m_svm$f1)
  
  ## COMPUTATION TIME ##
  comp_time <- microbenchmark(
    TSOLR = tsolr_2(y = train_data$Response, best_K = best_k, frequency = c(5, 100)), 
    ISOLR = vglm(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + Is.Seasonality3 + I0 + I1,
                 family = cumulative(link = "logitlink", parallel = ~ Is.Season1 + Is.Season2 + 
                                       Is.Seasonality2 + Is.Seasonality3 - 1, reverse = FALSE),
                 data = train_data), 
    MTD = mtd_fn(y = model_data$y_vec, y.training = train_data$y_vec, 
                 y.test = test_data$y_vec), 
    PAR = par_fn(y = model_data$y_vec, y.training = train_data$y_vec, 
                 y.test = test_data$y_vec), 
    RF = randomForest(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + 
                        Is.Seasonality3 + lag_y, data = train_data), 
    SVM = svm(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + 
                Is.Seasonality3 + lag_y, data = train_data, type = 'C', kernel = "sigmoid"), 
    times = 1
  )
  time_tsolr <- append(time_tsolr, summary(comp_time, unit = "ms")[1, 3])
  time_isolr <- append(time_isolr, summary(comp_time, unit = "ms")[2, 3])
  time_mtd <- append(time_mtd, summary(comp_time, unit = "ms")[3, 3])
  time_par <- append(time_par, summary(comp_time, unit = "ms")[4, 3])
  time_rf <- append(time_rf, summary(comp_time, unit = "ms")[5, 3])
  time_svm <- append(time_svm, summary(comp_time, unit = "ms")[6, 3])
}

results <- data.frame(
  Method = c("TSOLR", "ISOLR", "MTD", "PAR", "RF", "SVM"),
  
  Acc_Mean  = round(c(mean(acc_tsolr), mean(acc_isolr), mean(acc_mtd),
                      mean(acc_par), mean(acc_rf), mean(acc_svm)), 2),
  
  Acc_SD    = round(c(sdp(acc_tsolr), sdp(acc_isolr), sdp(acc_mtd),
                      sdp(acc_par), sdp(acc_rf), sdp(acc_svm)), 2),
  
  F1_Mean   = round(c(mean(f1_tsolr), mean(f1_isolr), mean(f1_mtd),
                      mean(f1_par), mean(f1_rf), mean(f1_svm)), 2),
  
  F1_SD     = round(c(sdp(f1_tsolr), sdp(f1_isolr), sdp(f1_mtd),
                      sdp(f1_par), sdp(f1_rf), sdp(f1_svm)), 2),
  
  Time_Mean = round(c(mean(time_tsolr), mean(time_isolr), mean(time_mtd),
                      mean(time_par), mean(time_rf), mean(time_svm)), 2),
  
  Time_SD   = round(c(sdp(time_tsolr), sdp(time_isolr), sdp(time_mtd),
                      sdp(time_par), sdp(time_rf), sdp(time_svm)), 2)
)

kable(results, align = "c", caption = "Performance comparison of methods")

Sys.time()

### K = 4 ###
Sys.time()
#set.seed(126)
# MODEL SPECIFICATION
theta0 <- -0.9
theta1 <- 0.02
theta2 <- 1.2
beta_season1 <- -0.4
beta_season2 <- -1.3
beta_seasonality2 <- 0.5
beta_seasonality3 <- 0.7
gamma00 <- -0.4
gamma01 <- 0.0
gamma02 <- 0.4
gamma10 <- -0.4
gamma11 <- 0.0
gamma12 <- 0.4
gamma20 <- -0.4
gamma21 <- 0.0
gamma22 <- 0.4
t_vec <- c(1:n)
Is.Season1 <- rep(replace(rep(0, 100), 1:30, 1), n_year)  #Season 1
Is.Season2 <- rep(replace(rep(0, 100), 31:70, 1), n_year)   #Season 2
Is.Seasonality2 <- rep(rep(replace(rep(0, 5), 4:5, 1), 20), n_year)   #Weekend
Is.Seasonality3 <- rep(replace(rep(0, 100), 81:85, 1), n_year)    #Diwali
lc <- (beta_season1 * Is.Season1) + (beta_season2 * Is.Season2) + 
  (beta_seasonality2 * Is.Seasonality2) + (beta_seasonality3 * Is.Seasonality3)

acc_tsolr <- c()
f1_tsolr <- c()
acc_isolr <- c()
f1_isolr <- c()
acc_mtd <- c()
f1_mtd <- c()
acc_par <- c()
f1_par <- c()
acc_rf <- c()
f1_rf <- c()
acc_svm <- c()
f1_svm <- c()
time_tsolr <- c()
time_isolr <- c()
time_mtd <- c()
time_par <- c()
time_rf <- c()
time_svm <- c()
for (iter in c(1:R)) {
  
  cat("Running Iteration", iter, "out of", R, "\n")
  
  ## DATA SIMULATION
  y_vec <- rep(0, (n + 1))
  y_vec[1] <- sample(c(0, 1, 2, 3), 1)
  for (t in c(2:(n + 1))) {
    I0 <- as.numeric(y_vec[t - 1] == 0)
    I1 <- as.numeric(y_vec[t - 1] == 1)
    I2 <- as.numeric(y_vec[t - 1] == 2) # CATEGORY 3 IS BASELINE
    eta0 <- lc[t - 1] + (gamma00 * I0) + (gamma01 * I1) + (gamma02 * I2)
    eta1 <- lc[t - 1] + (gamma10 * I0) + (gamma11 * I1) + (gamma12 * I2)
    eta2 <- lc[t - 1] + (gamma20 * I0) + (gamma21 * I1) + (gamma22 * I2)
    F0 <- plogis(theta0 - eta0)
    F1 <- plogis(theta1 - eta1)
    F2 <- plogis(theta2 - eta2)
    p0 <- F0
    p1 <- F1 - F0
    p2 <- F2 - F1
    p3 <- 1 - p0 - p1 - p2
    y_vec[t] <- sample(c(0, 1, 2, 3), size = 1, prob = c(p0, p1, p2, p3))
  }
  lag_y = y_vec[-length(y_vec)]
  
  model_data <- data.frame(t = t_vec, y_vec = y_vec[-1], 
                           Response = factor(y_vec[-1], levels = c(0, 1, 2, 3), ordered = T), 
                           Is.Season1, Is.Season2, Is.Seasonality2, Is.Seasonality3, 
                           lag_y = lag_y) %>% 
    mutate(I0 = as.numeric(lag_y == 0), I1 = as.numeric(lag_y == 1), I2 = as.numeric(lag_y == 2))
  
  # TRAIN-TEST SPLIT
  test_size <- n * 20 / 100
  train_data <- model_data[1:(n - test_size), ]
  test_data <- model_data[(n - (test_size - 1)):n, ]
  
  # TSOLR
  best_k <- tsolr_1(y = train_data$Response, num_seas = 2, frequency = c(5, 100))$best_K
  tsolr_model <- tsolr_2(y = train_data$Response, best_K = best_k, frequency = c(5, 100))
  pred_tsolr <- predict_tsolr(model = tsolr_model, train_y = train_data$Response, best_K = best_k,
                              test_y = test_data$Response, frequency = c(5, 100))$predicted_class
  m_tsolr <- compute_metrics(actual = test_data$Response, predicted = pred_tsolr, levels = 0:3)
  acc_tsolr <- append(acc_tsolr, m_tsolr$accuracy)
  f1_tsolr <- append(f1_tsolr, m_tsolr$f1)
  
  # ISOLR
  isolr_model <- vglm(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + Is.Seasonality3 + I0 + I1 + I2,
                      family = cumulative(link = "logitlink", parallel = ~ Is.Season1 + Is.Season2 + 
                                            Is.Seasonality2 + Is.Seasonality3 - 1, reverse = FALSE),
                      data = train_data)
  pred_isolr <- predict(isolr_model, newdata = test_data, type = "response")
  pred_isolr <- as.numeric(colnames(pred_isolr)[max.col(pred_isolr)])
  m_isolr <- compute_metrics(actual = test_data$Response, predicted = pred_isolr, levels = 0:3)
  acc_isolr <- append(acc_isolr, m_isolr$accuracy)
  f1_isolr <- append(f1_isolr, m_isolr$f1)
  
  # MTD
  pred_mtd <- mtd_fn(y = model_data$y_vec, y.training = train_data$y_vec, 
                     y.test = test_data$y_vec)
  m_mtd <- compute_metrics(actual = test_data$y_vec, predicted = pred_mtd, levels = 0:3)
  acc_mtd <- append(acc_mtd, m_mtd$accuracy)
  f1_mtd <- append(f1_mtd, m_mtd$f1)
  
  # PAR
  pred_par <- par_fn(y = model_data$y_vec, y.training = train_data$y_vec, 
                     y.test = test_data$y_vec)
  m_par <- compute_metrics(actual = test_data$y_vec, predicted = pred_par, levels = 0:3)
  acc_par <- append(acc_par, m_par$accuracy)
  f1_par <- append(f1_par, m_par$f1)
  
  # RF
  rf_model <- randomForest(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + 
                             Is.Seasonality3 + lag_y, data = train_data)
  pred_rf <- predict(rf_model, test_data)
  m_rf <- compute_metrics(actual = test_data$Response, predicted = pred_rf, levels = 0:3)
  acc_rf <- append(acc_rf, m_rf$accuracy)
  f1_rf <- append(f1_rf, m_rf$f1)
  
  # SVM
  svm_model <- svm(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + 
                     Is.Seasonality3 + lag_y, data = train_data, type = 'C', kernel = "sigmoid")
  pred_svm <- predict(svm_model, test_data)
  m_svm <- compute_metrics(actual = test_data$Response, predicted = pred_svm, levels = 0:3)
  acc_svm <- append(acc_svm, m_svm$accuracy)
  f1_svm <-append(f1_svm, m_svm$f1)
  
  ## COMPUTATION TIME ##
  comp_time <- microbenchmark(
    TSOLR = tsolr_2(y = train_data$Response, best_K = best_k, frequency = c(5, 100)), 
    ISOLR = vglm(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + Is.Seasonality3 + I0 + I1 + I2,
                 family = cumulative(link = "logitlink", parallel = ~ Is.Season1 + Is.Season2 + 
                                       Is.Seasonality2 + Is.Seasonality3 - 1, reverse = FALSE),
                 data = train_data), 
    MTD = mtd_fn(y = model_data$y_vec, y.training = train_data$y_vec, 
                 y.test = test_data$y_vec), 
    PAR = par_fn(y = model_data$y_vec, y.training = train_data$y_vec, 
                 y.test = test_data$y_vec), 
    RF = randomForest(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + 
                        Is.Seasonality3 + lag_y, data = train_data), 
    SVM = svm(Response ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + 
                Is.Seasonality3 + lag_y, data = train_data, type = 'C', kernel = "sigmoid"), 
    times = 1
  )
  time_tsolr <- append(time_tsolr, summary(comp_time, unit = "ms")[1, 3])
  time_isolr <- append(time_isolr, summary(comp_time, unit = "ms")[2, 3])
  time_mtd <- append(time_mtd, summary(comp_time, unit = "ms")[3, 3])
  time_par <- append(time_par, summary(comp_time, unit = "ms")[4, 3])
  time_rf <- append(time_rf, summary(comp_time, unit = "ms")[5, 3])
  time_svm <- append(time_svm, summary(comp_time, unit = "ms")[6, 3])
}

results <- data.frame(
  Method = c("TSOLR", "ISOLR", "MTD", "PAR", "RF", "SVM"),
  
  Acc_Mean  = round(c(mean(acc_tsolr), mean(acc_isolr), mean(acc_mtd),
                      mean(acc_par), mean(acc_rf), mean(acc_svm)), 2),
  
  Acc_SD    = round(c(sdp(acc_tsolr), sdp(acc_isolr), sdp(acc_mtd),
                      sdp(acc_par), sdp(acc_rf), sdp(acc_svm)), 2),
  
  F1_Mean   = round(c(mean(f1_tsolr), mean(f1_isolr), mean(f1_mtd),
                      mean(f1_par), mean(f1_rf), mean(f1_svm)), 2),
  
  F1_SD     = round(c(sdp(f1_tsolr), sdp(f1_isolr), sdp(f1_mtd),
                      sdp(f1_par), sdp(f1_rf), sdp(f1_svm)), 2),
  
  Time_Mean = round(c(mean(time_tsolr), mean(time_isolr), mean(time_mtd),
                      mean(time_par), mean(time_rf), mean(time_svm)), 2),
  
  Time_SD   = round(c(sdp(time_tsolr), sdp(time_isolr), sdp(time_mtd),
                      sdp(time_par), sdp(time_rf), sdp(time_svm)), 2)
)

kable(results, align = "c", caption = "Performance comparison of methods")

Sys.time()

