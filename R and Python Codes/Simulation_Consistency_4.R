library(tidyverse)
library(VGAM)

sdp <- function(x) {sqrt(mean((x - mean(x)) ^ 2))}

R <-  1000

n_year <- 5   #5, 10, 100
n_dpy <- 100   #number of days in each year block
n <- n_year * n_dpy

##### CONSISTENCY OF TSOLR #####
### K = 3 ###
# model specification
theta0 <- -0.84
theta1 <- 0.84
beta_sin_1 <- 0   #season
beta_cos_1 <- 2   #season
beta_sin_2 <- 0   #diwali
beta_cos_2 <- 0.7 #diwali
beta_sin_3 <- 0   #daytype
beta_cos_3 <- 0.2 #daytype
gamma00 <- -1.0
gamma01 <-  0.5
gamma10 <- -0.8
gamma11 <-  0.3
t_vec <- c(1:n)
sin_1 <- sin(2 * pi * t_vec / 100)  #season
cos_1 <- cos(2 * pi * t_vec / 100)  #season
sin_2 <- sin(2 * pi * t_vec / 25)   #diwali
cos_2 <- cos(2 * pi * t_vec / 25)   #diwali
sin_3 <- sin(2 * pi * t_vec / 5)    #daytype
cos_3 <- cos(2 * pi * t_vec / 5)    #daytype

# model fitting
theta0_vec <- c()
theta1_vec <- c()
sin_1_vec <- c()
cos_1_vec <- c()
sin_2_vec <- c()
cos_2_vec <- c()
sin_3_vec <- c()
cos_3_vec <- c()
gamma00_vec <- c()
gamma01_vec <- c()
gamma10_vec <- c()
gamma11_vec <- c()
for (r in c(1:R)) {
  
  cat("Running Iteration", r, "out of", R, "\n")
  
  ## DATA SIMULATION
  y_vec <- rep(0, n)
  y_vec[1] <- sample(c(0, 1, 2), 1)
  for (t in c(2:n)) {
    I0 <- as.numeric(y_vec[t - 1] == 0)
    I1 <- as.numeric(y_vec[t - 1] == 1) # CATEGORY 2 IS BASELINE
    eta0 <- (beta_sin_1 * sin_1[t]) + (beta_cos_1 * cos_1[t]) + (beta_sin_2 * sin_2[t]) + 
      (beta_cos_2 * cos_2[t]) + (beta_sin_3 * sin_3[t]) + (beta_cos_3 * cos_3[t]) + 
      (gamma00 * I0) + (gamma01 * I1)
    eta1 <- (beta_sin_1 * sin_1[t]) + (beta_cos_1 * cos_1[t]) + (beta_sin_2 * sin_2[t]) + 
      (beta_cos_2 * cos_2[t]) + (beta_sin_3 * sin_3[t]) + (beta_cos_3 * cos_3[t]) + 
      (gamma10 * I0) + (gamma11 * I1)
    F0 <- plogis(theta0 - eta0)
    F1 <- plogis(theta1 - eta1)
    p0 <- F0
    p1 <- F1 - F0
    p2 <- 1 - p0 - p1
    y_vec[t] <- sample(c(0, 1, 2), size = 1, prob = c(p0, p1, p2))
  }
  
  ## LAG CREATION
  lag_y <- c(NA, y_vec[-n])
  I0 <- as.numeric(lag_y == 0)
  I1 <- as.numeric(lag_y == 1)
  
  sim_data <- data.frame(y_vec = factor(y_vec, levels = c(0, 1, 2), ordered = TRUE), 
                         sin_1, cos_1, sin_2, cos_2, sin_3, cos_3, I0, I1)
  sim_data <- sim_data[-1, ]
  
  ## MODEL FITTING
  log_model <- vglm(y_vec ~ sin_1 + cos_1 + sin_2 + cos_2 + sin_3 + cos_3 + I0 + I1, 
                    family = cumulative(link = "logitlink", parallel = TRUE ~ sin_1 + 
                                          cos_1 + sin_2 + cos_2 + sin_3 + cos_3 - 1), 
                    data = sim_data)
  
  ## VALUE COLLECTION
  cf <- coef(log_model)
  theta0_vec <- append(theta0_vec, cf["(Intercept):1"])
  theta1_vec <- append(theta1_vec, cf["(Intercept):2"])
  sin_1_vec <- append(sin_1_vec, cf["sin_1"])
  cos_1_vec <- append(cos_1_vec, cf["cos_1"])
  sin_2_vec <- append(sin_2_vec, cf["sin_2"])
  cos_2_vec <- append(cos_2_vec, cf["cos_2"])
  sin_3_vec <- append(sin_3_vec, cf["sin_3"])
  cos_3_vec <- append(cos_3_vec, cf["cos_3"])
  gamma00_vec <- append(gamma00_vec, cf["I0:1"])
  gamma10_vec <- append(gamma10_vec, cf["I0:2"])
  gamma01_vec <- append(gamma01_vec, cf["I1:1"])
  gamma11_vec <- append(gamma11_vec, cf["I1:2"])
}

# performance checking
round(c(mean(theta0_vec), sdp(theta0_vec)), 2)
round(c(mean(theta1_vec), sdp(theta1_vec)), 2)
round(c(mean(-sin_1_vec), sdp(-sin_1_vec)), 2)
round(c(mean(-cos_1_vec), sdp(-cos_1_vec)), 2)
round(c(mean(-sin_2_vec), sdp(-sin_2_vec)), 2)
round(c(mean(-cos_2_vec), sdp(-cos_2_vec)), 2)
round(c(mean(-sin_3_vec), sdp(-sin_3_vec)), 2)
round(c(mean(-cos_3_vec), sdp(-cos_3_vec)), 2)
round(c(mean(-gamma00_vec), sdp(-gamma00_vec)), 2)
round(c(mean(-gamma01_vec), sdp(-gamma01_vec)), 2)
round(c(mean(-gamma10_vec), sdp(-gamma10_vec)), 2)
round(c(mean(-gamma11_vec), sdp(-gamma11_vec)), 2)



### K = 4 ###
# model specification
theta0 <- -1.1
theta1 <- 0
theta2 <- 1.9
beta_sin_1 <- 0.6   #season
beta_cos_1 <- -1.1  #season
beta_sin_2 <- 1.4   #diwali
beta_cos_2 <- -0.8  #diwali
beta_sin_3 <- -0.1  #daytype
beta_cos_3 <- -0.4  #daytype
gamma00 <- -1.2
gamma01 <- -0.5
gamma02 <-  0.3
gamma10 <- -0.8
gamma11 <- -0.3
gamma12 <-  0.2
gamma20 <- -0.4
gamma21 <- -0.1
gamma22 <-  0.1
t_vec <- c(1:n)
sin_1 <- sin(2 * pi * t_vec / 100)  #season
cos_1 <- cos(2 * pi * t_vec / 100)  #season
sin_2 <- sin(2 * pi * t_vec / 25)   #diwali
cos_2 <- cos(2 * pi * t_vec / 25)   #diwali
sin_3 <- sin(2 * pi * t_vec / 5)    #daytype
cos_3 <- cos(2 * pi * t_vec / 5)    #daytype

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
gamma00_vec <- c()
gamma01_vec <- c()
gamma02_vec <- c()
gamma10_vec <- c()
gamma11_vec <- c()
gamma12_vec <- c()
gamma20_vec <- c()
gamma21_vec <- c()
gamma22_vec <- c()
for (r in c(1:1000)) {
  
  cat("Running Iteration", r, "out of", R, "\n")
  
  ## DATA SIMULATION
  y_vec <- rep(0, n)
  y_vec[1] <- sample(c(0, 1, 2, 3), 1)
  for (t in c(2:n)) {
    I0 <- as.numeric(y_vec[t - 1] == 0)
    I1 <- as.numeric(y_vec[t - 1] == 1)
    I2 <- as.numeric(y_vec[t - 1] == 2) # CATEGORY 3 IS BASELINE
    eta0 <- (beta_sin_1 * sin_1[t]) + (beta_cos_1 * cos_1[t]) + (beta_sin_2 * sin_2[t]) + 
      (beta_cos_2 * cos_2[t]) + (beta_sin_3 * sin_3[t]) + (beta_cos_3 * cos_3[t]) + 
      (gamma00 * I0) + (gamma01 * I1) + (gamma02 * I2)
    eta1 <- (beta_sin_1 * sin_1[t]) + (beta_cos_1 * cos_1[t]) + (beta_sin_2 * sin_2[t]) + 
      (beta_cos_2 * cos_2[t]) + (beta_sin_3 * sin_3[t]) + (beta_cos_3 * cos_3[t]) + 
      (gamma10 * I0) + (gamma11 * I1) + (gamma12 * I2)
    eta2 <- (beta_sin_1 * sin_1[t]) + (beta_cos_1 * cos_1[t]) + (beta_sin_2 * sin_2[t]) + 
      (beta_cos_2 * cos_2[t]) + (beta_sin_3 * sin_3[t]) + (beta_cos_3 * cos_3[t]) + 
      (gamma20 * I0) + (gamma21 * I1) + (gamma22 * I2)
    F0 <- plogis(theta0 - eta0)
    F1 <- plogis(theta1 - eta1)
    F2 <- plogis(theta2 - eta2)
    p0 <- F0
    p1 <- F1 - F0
    p2 <- F2 - F1
    p3 <- 1 - p0 - p1 - p2
    y_vec[t] <- sample(c(0, 1, 2, 3), size = 1, prob = c(p0, p1, p2, p3))
  }
  
  ## LAG CREATION
  lag_y <- c(NA, y_vec[-n])
  I0 <- as.numeric(lag_y == 0)
  I1 <- as.numeric(lag_y == 1)
  I2 <- as.numeric(lag_y == 2)
  
  sim_data <- data.frame(y_vec = factor(y_vec, levels = c(0, 1, 2, 3), ordered = TRUE), 
                         sin_1, cos_1, sin_2, cos_2, sin_3, cos_3, I0, I1, I2)
  sim_data <- sim_data[-1, ]
  
  ## MODEL FITTING
  log_model <- vglm(y_vec ~ sin_1 + cos_1 + sin_2 + cos_2 + sin_3 + cos_3 + I0 + I1 + I2, 
                    family = cumulative(link = "logitlink", parallel = TRUE ~ sin_1 + 
                                          cos_1 + sin_2 + cos_2 + sin_3 + cos_3 - 1), 
                    data = sim_data)
  
  ## VALUE COLLECTION
  cf <- coef(log_model)
  theta0_vec <- append(theta0_vec, cf["(Intercept):1"])
  theta1_vec <- append(theta1_vec, cf["(Intercept):2"])
  theta2_vec <- append(theta2_vec, cf["(Intercept):3"])
  sin_1_vec <- append(sin_1_vec, cf["sin_1"])
  cos_1_vec <- append(cos_1_vec, cf["cos_1"])
  sin_2_vec <- append(sin_2_vec, cf["sin_2"])
  cos_2_vec <- append(cos_2_vec, cf["cos_2"])
  sin_3_vec <- append(sin_3_vec, cf["sin_3"])
  cos_3_vec <- append(cos_3_vec, cf["cos_3"])
  gamma00_vec <- append(gamma00_vec, cf["I0:1"])
  gamma10_vec <- append(gamma10_vec, cf["I0:2"])
  gamma20_vec <- append(gamma20_vec, cf["I0:3"])
  gamma01_vec <- append(gamma01_vec, cf["I1:1"])
  gamma11_vec <- append(gamma11_vec, cf["I1:2"])
  gamma21_vec <- append(gamma21_vec, cf["I1:3"])
  gamma02_vec <- append(gamma02_vec, cf["I2:1"])
  gamma12_vec <- append(gamma12_vec, cf["I2:2"])
  gamma22_vec <- append(gamma22_vec, cf["I2:3"])
}

# performance checking
round(c(mean(theta0_vec), sdp(theta0_vec)), 2)
round(c(mean(theta1_vec), sdp(theta1_vec)), 2)
round(c(mean(theta2_vec), sdp(theta2_vec)), 2)
round(c(mean(-sin_1_vec), sdp(-sin_1_vec)), 2)
round(c(mean(-cos_1_vec), sdp(-cos_1_vec)), 2)
round(c(mean(-sin_2_vec), sdp(-sin_2_vec)), 2)
round(c(mean(-cos_2_vec), sdp(-cos_2_vec)), 2)
round(c(mean(-sin_3_vec), sdp(-sin_3_vec)), 2)
round(c(mean(-cos_3_vec), sdp(-cos_3_vec)), 2)
round(c(mean(-gamma00_vec), sdp(-gamma00_vec)), 2)
round(c(mean(-gamma01_vec), sdp(-gamma01_vec)), 2)
round(c(mean(-gamma02_vec), sdp(-gamma02_vec)), 2)
round(c(mean(-gamma10_vec), sdp(-gamma10_vec)), 2)
round(c(mean(-gamma11_vec), sdp(-gamma11_vec)), 2)
round(c(mean(-gamma12_vec), sdp(-gamma12_vec)), 2)
round(c(mean(-gamma20_vec), sdp(-gamma20_vec)), 2)
round(c(mean(-gamma21_vec), sdp(-gamma21_vec)), 2)
round(c(mean(-gamma22_vec), sdp(-gamma22_vec)), 2)





##### CONSISTENCY OF ISOLR #####
### K = 3 ###
# model specification
theta0 <- -0.78
theta1 <- 0.75
beta_season1 <- -0.3
beta_season2 <- 1.2
beta_seasonality2 <- -0.4
beta_seasonality3 <- -0.7
gamma00 <- 0.7
gamma01 <- 0.1
gamma10 <- -0.2
gamma11 <- 0.6
t_vec <- c(1:n)
Is.Season1 <- rep(replace(rep(0, 100), 1:30, 1), n_year)  #Season 1
Is.Season2 <- rep(replace(rep(0, 100), 31:70, 1), n_year)   #Season 2
Is.Seasonality2 <- rep(rep(replace(rep(0, 5), 4:5, 1), 20), n_year)   #Weekend
Is.Seasonality3 <- rep(replace(rep(0, 100), 81:85, 1), n_year)    #Diwali
lc <- (beta_season1 * Is.Season1) + (beta_season2 * Is.Season2) + 
  (beta_seasonality2 * Is.Seasonality2) + (beta_seasonality3 * Is.Seasonality3)

# model fitting
theta0_vec <- c()
theta1_vec <- c()
beta_season1_vec <- c()
beta_season2_vec <- c()
beta_seasonality2_vec <- c()
beta_seasonality3_vec <- c()
gamma00_vec <- c()
gamma01_vec <- c()
gamma10_vec <- c()
gamma11_vec <- c()
for (r in c(1:R)) {
  
  cat("Running Iteration", r, "out of", R, "\n")
  
  ## DATA SIMULATION
  y_vec <- rep(0, n)
  y_vec[1] <- sample(c(0, 1, 2), 1)
  for (t in c(2:n)) {
    I0 <- as.numeric(y_vec[t - 1] == 0)
    I1 <- as.numeric(y_vec[t - 1] == 1) # CATEGORY 2 IS BASELINE
    eta0 <- lc[t] + (gamma00 * I0) + (gamma01 * I1)
    eta1 <- lc[t] + (gamma10 * I0) + (gamma11 * I1)
    F0 <- plogis(theta0 - eta0)
    F1 <- plogis(theta1 - eta1)
    p0 <- F0
    p1 <- F1 - F0
    p2 <- 1 - p0 - p1
    y_vec[t] <- sample(c(0, 1, 2), size = 1, prob = c(p0, p1, p2))
  }
  
  ## LAG CREATION
  lag_y <- c(NA, y_vec[-n])
  I0 <- as.numeric(lag_y == 0)
  I1 <- as.numeric(lag_y == 1)
  
  sim_data <- data.frame(y_vec = factor(y_vec, levels = c(0, 1, 2), ordered = TRUE), 
                         Is.Season1, Is.Season2, Is.Seasonality2, Is.Seasonality3, I0, I1)
  sim_data <- sim_data[-1, ]
  
  ## MODEL FITTING
  log_model <- vglm(y_vec ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + Is.Seasonality3 + I0 + I1, 
                    family = cumulative(link = "logitlink", parallel = TRUE ~ Is.Season1 + Is.Season2 + 
                                          Is.Seasonality2 + Is.Seasonality3 - 1), 
                    data = sim_data)
  
  ## VALUE COLLECTION
  cf <- coef(log_model)
  theta0_vec <- append(theta0_vec, cf["(Intercept):1"])
  theta1_vec <- append(theta1_vec, cf["(Intercept):2"])
  beta_season1_vec <- append(beta_season1_vec, cf["Is.Season1"])
  beta_season2_vec <- append(beta_season2_vec, cf["Is.Season2"])
  beta_seasonality2_vec <- append(beta_seasonality2_vec, cf["Is.Seasonality2"])
  beta_seasonality3_vec <- append(beta_seasonality3_vec, cf["Is.Seasonality3"])
  gamma00_vec <- append(gamma00_vec, cf["I0:1"])
  gamma10_vec <- append(gamma10_vec, cf["I0:2"])
  gamma01_vec <- append(gamma01_vec, cf["I1:1"])
  gamma11_vec <- append(gamma11_vec, cf["I1:2"])
}

# performance checking
round(c(mean(theta0_vec), sdp(theta0_vec)), 2)
round(c(mean(theta1_vec), sdp(theta1_vec)), 2)
round(c(mean(-beta_season1_vec), sdp(-beta_season1_vec)), 2)
round(c(mean(-beta_season2_vec), sdp(-beta_season2_vec)), 2)
round(c(mean(-beta_seasonality2_vec), sdp(-beta_seasonality2_vec)), 2)
round(c(mean(-beta_seasonality3_vec), sdp(-beta_seasonality3_vec)), 2)
round(c(mean(-gamma00_vec), sdp(-gamma00_vec)), 2)
round(c(mean(-gamma01_vec), sdp(-gamma01_vec)), 2)
round(c(mean(-gamma10_vec), sdp(-gamma10_vec)), 2)
round(c(mean(-gamma11_vec), sdp(-gamma11_vec)), 2)



### K = 4 ###
# model specification
theta0 <- -1
theta1 <- -0.03
theta2 <- 1.5
beta_season1 <- 0.3
beta_season2 <- 1.4
beta_seasonality2 <- -0.3
beta_seasonality3 <- -0.5
gamma00 <- 0.30
gamma01 <- 0.1
gamma02 <- -0.30
gamma10 <- 0.20
gamma11 <- 0.3
gamma12 <- -0.20
gamma20 <- 0.10
gamma21 <- -0.2
gamma22 <- -0.10
t_vec <- c(1:n)
Is.Season1 <- rep(replace(rep(0, 100), 1:30, 1), n_year)  #Season 1
Is.Season2 <- rep(replace(rep(0, 100), 31:70, 1), n_year)   #Season 2
Is.Seasonality2 <- rep(rep(replace(rep(0, 5), 4:5, 1), 20), n_year)   #Weekend
Is.Seasonality3 <- rep(replace(rep(0, 100), 81:85, 1), n_year)    #Diwali
lc <- (beta_season1 * Is.Season1) + (beta_season2 * Is.Season2) + 
  (beta_seasonality2 * Is.Seasonality2) + (beta_seasonality3 * Is.Seasonality3)

# model fitting
theta0_vec <- c()
theta1_vec <- c()
theta2_vec <- c()
beta_season1_vec <- c()
beta_season2_vec <- c()
beta_seasonality2_vec <- c()
beta_seasonality3_vec <- c()
gamma00_vec <- c()
gamma01_vec <- c()
gamma02_vec <- c()
gamma10_vec <- c()
gamma11_vec <- c()
gamma12_vec <- c()
gamma20_vec <- c()
gamma21_vec <- c()
gamma22_vec <- c()
for (r in c(1:R)) {
  
  cat("Running Iteration", r, "out of", R, "\n")
  
  ## DATA SIMULATION
  y_vec <- rep(0, n)
  y_vec[1] <- sample(c(0, 1, 2, 3), 1)
  for (t in c(2:n)) {
    I0 <- as.numeric(y_vec[t - 1] == 0)
    I1 <- as.numeric(y_vec[t - 1] == 1)
    I2 <- as.numeric(y_vec[t - 1] == 2) # CATEGORY 3 IS BASELINE
    eta0 <- lc[t] + (gamma00 * I0) + (gamma01 * I1) + (gamma02 * I2)
    eta1 <- lc[t] + (gamma10 * I0) + (gamma11 * I1) + (gamma12 * I2)
    eta2 <- lc[t] + (gamma20 * I0) + (gamma21 * I1) + (gamma22 * I2)
    F0 <- plogis(theta0 - eta0)
    F1 <- plogis(theta1 - eta1)
    F2 <- plogis(theta2 - eta2)
    p0 <- F0
    p1 <- F1 - F0
    p2 <- F2 - F1
    p3 <- 1 - p0 - p1 - p2
    y_vec[t] <- sample(c(0, 1, 2, 3), size = 1, prob = c(p0, p1, p2, p3))
  }
  
  ## LAG CREATION
  lag_y <- c(NA, y_vec[-n])
  I0 <- as.numeric(lag_y == 0)
  I1 <- as.numeric(lag_y == 1)
  I2 <- as.numeric(lag_y == 2)
  
  sim_data <- data.frame(y_vec = factor(y_vec, levels = c(0, 1, 2, 3), ordered = TRUE), 
                         Is.Season1, Is.Season2, Is.Seasonality2, Is.Seasonality3, I0, I1, I2)
  sim_data <- sim_data[-1, ]
  
  ## MODEL FITTING
  log_model <- vglm(y_vec ~ Is.Season1 + Is.Season2 + Is.Seasonality2 + Is.Seasonality3 + I0 + I1 + I2, 
                    family = cumulative(link = "logitlink", parallel = TRUE ~ Is.Season1 + Is.Season2 + 
                                          Is.Seasonality2 + Is.Seasonality3 - 1), 
                    data = sim_data)
  
  ## VALUE COLLECTION
  cf <- coef(log_model)
  theta0_vec <- append(theta0_vec, cf["(Intercept):1"])
  theta1_vec <- append(theta1_vec, cf["(Intercept):2"])
  theta2_vec <- append(theta2_vec, cf["(Intercept):3"])
  beta_season1_vec <- append(beta_season1_vec, cf["Is.Season1"])
  beta_season2_vec <- append(beta_season2_vec, cf["Is.Season2"])
  beta_seasonality2_vec <- append(beta_seasonality2_vec, cf["Is.Seasonality2"])
  beta_seasonality3_vec <- append(beta_seasonality3_vec, cf["Is.Seasonality3"])
  gamma00_vec <- append(gamma00_vec, cf["I0:1"])
  gamma10_vec <- append(gamma10_vec, cf["I0:2"])
  gamma20_vec <- append(gamma20_vec, cf["I0:3"])
  gamma01_vec <- append(gamma01_vec, cf["I1:1"])
  gamma11_vec <- append(gamma11_vec, cf["I1:2"])
  gamma21_vec <- append(gamma21_vec, cf["I1:3"])
  gamma02_vec <- append(gamma02_vec, cf["I2:1"])
  gamma12_vec <- append(gamma12_vec, cf["I2:2"])
  gamma22_vec <- append(gamma22_vec, cf["I2:3"])
}

# performance checking
round(c(mean(theta0_vec), sdp(theta0_vec)), 2)
round(c(mean(theta1_vec), sdp(theta1_vec)), 2)
round(c(mean(theta2_vec), sdp(theta2_vec)), 2)
round(c(mean(-beta_season1_vec), sdp(-beta_season1_vec)), 2)
round(c(mean(-beta_season2_vec), sdp(-beta_season2_vec)), 2)
round(c(mean(-beta_seasonality2_vec), sdp(-beta_seasonality2_vec)), 2)
round(c(mean(-beta_seasonality3_vec), sdp(-beta_seasonality3_vec)), 2)
round(c(mean(-gamma00_vec), sdp(-gamma00_vec)), 2)
round(c(mean(-gamma01_vec), sdp(-gamma01_vec)), 2)
round(c(mean(-gamma02_vec), sdp(-gamma02_vec)), 2)
round(c(mean(-gamma10_vec), sdp(-gamma10_vec)), 2)
round(c(mean(-gamma11_vec), sdp(-gamma11_vec)), 2)
round(c(mean(-gamma12_vec), sdp(-gamma12_vec)), 2)
round(c(mean(-gamma20_vec), sdp(-gamma20_vec)), 2)
round(c(mean(-gamma21_vec), sdp(-gamma21_vec)), 2)
round(c(mean(-gamma22_vec), sdp(-gamma22_vec)), 2)
