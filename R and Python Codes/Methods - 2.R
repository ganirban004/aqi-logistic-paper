library(schoolmath)
library(MASS)
library(NlcOptim)
library(matrixcalc)
library(tidyverse)
library(randomForest)
library(e1071)
library(patchwork)
library(ordinal)
library(forecast)
library(VGAM)

mode.pred = function(pred.dist){
  a=c(0:(length(pred.dist)-1))
  return(min(a[which(pred.dist==max(pred.dist))]))
}

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

# MTD
mtd_fn <- function(y, y.training, y.test) {
  tr=max(y)
  k=tr+1
  
  n_s=length(y.training)
  
  kk=max(y.training)+1
  
  ### preparation of likelihood functions
  
  J=matrix(0, kk,1)
  for(i in 1:kk){
    for(t in 1:n_s){
      if(y.training[t]==(i-1))
        J[i]=J[i]+1
    }
  }
  
  ### preparation for MTD forecasting
  
  # calculating Q
  
  N=matrix(0,ncol=(tr+1),nrow=(tr+1))
  Q=matrix(0,ncol=(tr+1),nrow=(tr+1))
  le=length(y.training)
  p=y.training[1:(le-1)]
  q=y.training[2:le]
  for(j in 0:tr)
  {
    for(k in 0:tr){
      cou=0
      for(i in 1:(le-1))
      {
        if(p[i]==j&&q[i]==k)
          cou=cou+1
        else
          cou=cou
      }
      N[j+1,k+1]=cou
    }
  }
  
  for(j in 0:tr){
    for(k in 0:tr){
      if(sum(N[j+1,])==0)
        Q[j+1,k+1]=N[j+1,k+1]
      else
        Q[j+1,k+1]=N[j+1,k+1]/sum(N[j+1,])  
    }
  }
  
  predMTD=function(uu,h){
    ma=matrix.power(Q,h)
    pp=NULL
    for(i in 0:tr){
      pp[i+1]=ma[uu+1,i+1]    
    }
    return(pp)
  }
  
  # predicted error for a single data set
  
  predictedErrorMTD = function(u, training.set, test.set){ 
    n = length(training.set)
    m = length(test.set)
    
    predictedErrorMTD_h = function(u, h, test.set){         
      pred.dist = sapply(1:(m-h+1), function(j) predMTD(u[n+j-1], h))
      mode = sapply(1:(m-h+1), function(j) mode.pred(pred.dist[,j]))
      return(c(ptp(test.set[h:m], mode)))  
    }      
    
    return( t(sapply(1:3, function(h) 
      predictedErrorMTD_h(u, h, test.set))) )
  }
  
  # predicted values for a single data set
  predictedModeMTD = function(u, training.set, test.set) { 
    n = length(training.set)
    m = length(test.set)
    
    predictedModeMTD_h = function(u, h, test.set){         
      pred.dist = sapply(1:(m - h + 1), function(j) predMTD(u[n + j - 1], h))
      mode = sapply(1:(m - h + 1), function(j) mode.pred(pred.dist[, j]))
      return(mode)
    }      
    
    # Return modes for h = 1, 2, 3
    mode_matrix <- lapply(1:3, function(h) predictedModeMTD_h(u, h, test.set))
    names(mode_matrix) <- paste0("h", 1:3)
    return(mode_matrix)
  }
  
  pred_MTD <- predictedModeMTD(y, y.training, y.test)$h1
  return(pred_MTD)
}

# PAR
par_fn <- function(y, y.training, y.test) {
  tr=max(y)
  k=tr+1
  
  
  g1=function(a,b) #PAR
  {
    if(a==b)
      return(1)
    else
      return(0)
  }
  
  n=length(y) # sample size
  
  
  # Yt's marginal
  
  id=diag(1,nrow=max(y)+1,ncol=max(y)+1)
  
  n_s=length(y.training)
  
  kk=max(y.training)+1
  
  ### preparation of likelihood functions
  
  J=matrix(0, kk,1)
  for(i in 1:kk){
    for(t in 1:n_s){
      if(y.training[t]==(i-1))
        J[i]=J[i]+1
    }
  }
  
  
  # initial value for par process
  
  ph.start=abs(cor(y.training[1:length(y.training)-1],
                   y.training[2:length(y.training)]))
  
  ### preparation for PAR forecasting
  
  # h-step ahead conditional distribution p_h(j|yn)
  
  conditionalPAR = function(ynh, yn, h, phi, p_v){
    q = phi^h * g1(ynh,yn) + (1-phi^h) * p_v[ynh+1]
    return(q)
  }
  
  # mle function
  
  mlePAR=function(yy){
    n=length(yy)
    likelihood_par=function(xx){
      phi_hat=xx[1]
      f=log(J[yy[1]+1]/n)
      
      for(i in 1:(n-1)){
        f=f+log(phi_hat*g1(yy[i+1],yy[i])+(1-phi_hat)*(J[yy[i+1]+1]/n))
      }
      return(-f)
    }
    opt2=optim(fn=likelihood_par, par=c(ph.start),
               lower=c(0.01), upper=c(0.99), method="L-BFGS-B")
    return(opt2$par)
  }
  
  # distribution of Y_(n+h) given Y_n = (p_h(0), p_h(1), ..., p_h(l))
  
  predPAR = function(yn, h, pp){
    phi_hat=pp[1]
    J_eps=J/n_s
    l=tr
    return(sapply(0:l, function(j) conditionalPAR(j, yn, h, phi_hat, J_eps)))
  }
  
  # predicted error for a single data set
  
  predictedErrorPAR = function(u, training.set, test.set){
    n = length(training.set)
    m = length(test.set)
    
    predictedErrorPAR_h = function(u, h, beta, test.set){
      pred.dist = sapply(1:(m-h+1), function(j) predPAR(u[n+j-1], h, beta))
      mode = sapply(1:(m-h+1), function(j) mode.pred(pred.dist[,j]))
      return(c(ptp(test.set[h:m], mode)))
    }
    
    mle = mlePAR(training.set)
    return( t(sapply(1:3, function(h)
      predictedErrorPAR_h(u, h, mle, test.set))) )
  }
  
  # predicted values for a single data set
  predictedModePAR = function(u, training.set, test.set) {
    n = length(training.set)
    m = length(test.set)
    
    predictedModePAR_h = function(u, h, beta, test.set) {
      pred.dist = sapply(1:(m - h + 1), function(j) predPAR(u[n + j - 1], h, beta))
      mode = sapply(1:(m - h + 1), function(j) mode.pred(pred.dist[, j]))
      return(mode)
    }
    
    # Estimate parameters using training data
    mle = mlePAR(training.set)
    
    # Compute mode predictions for h = 1, 2, 3
    mode_list <- lapply(1:3, function(h) predictedModePAR_h(u, h, mle, test.set))
    names(mode_list) <- paste0("h", 1:3)
    
    return(mode_list)
  }
  
  pred_PAR <- predictedModePAR(y, y.training, y.test)$h1
  return(pred_PAR)
}

# RF

# SVM

# ISOLR

# TSOLR
tsolr_1 <- function(y, num_seas, frequency) {
  
  if (length(frequency) != num_seas) {
    stop("length(frequency) must equal num_seas")
  }
  
  y <- factor(y, ordered = TRUE)
  n <- length(y)
  
  lag_y <- factor(c(NA, head(y, -1)), levels = levels(y))
  
  tt <- seq_len(n)
  
  fourier_list <- vector("list", num_seas)
  
  for (i in seq_len(num_seas)) {
    Pi <- frequency[i]
    Ki_max <- 10
    tmp <- data.frame(matrix(nrow = n, ncol = 0))
    
    for (k in seq_len(Ki_max)) {
      tmp[[paste0("cos_", i, "_", k)]] <-
        cos(2 * pi * k * tt / Pi)
      tmp[[paste0("sin_", i, "_", k)]] <-
        sin(2 * pi * k * tt / Pi)
    }
    
    fourier_list[[i]] <- tmp
  }
  
  K_grid <- lapply(frequency, function(Pi) {seq_len(Ki_max)})
  K_combinations <- expand.grid(K_grid)
  
  best_aic <- Inf
  best_model <- NULL
  best_K <- NULL
  
  # NEW: store AIC for every combination
  aic_table <- K_combinations
  aic_table$AIC <- NA_real_
  
  ##################################################
  # Grid search
  ##################################################
  
  for (r in seq_len(nrow(K_combinations))) {
    
    cat("Running Iteration", r, "out of", nrow(K_combinations), "\n")
    
    current_K <- as.numeric(K_combinations[r, ])
    X <- data.frame(matrix(nrow = n, ncol = 0))
    
    for (i in seq_len(num_seas)) {
      Ki <- current_K[i]
      cols_needed <- c()
      
      for (k in seq_len(Ki)) {
        cols_needed <- c(
          cols_needed,
          paste0("cos_", i, "_", k),
          paste0("sin_", i, "_", k)
        )
      }
      
      X <- cbind(
        X,
        fourier_list[[i]][, cols_needed, drop = FALSE]
      )
    }
    
    dat <- data.frame(y = y, lag_y = lag_y, X)
    dat <- dat[-1, , drop = FALSE]
    
    form <- as.formula(
      paste("y ~", paste(colnames(dat)[-1], collapse = " + "))
    )
    
    ################################################
    # Fit model
    ################################################
    
    fit <- tryCatch({
      suppressWarnings(
        VGAM::vglm(
          formula = form,
          family = VGAM::cumulative(
            parallel = TRUE,
            reverse = FALSE
          ),
          data = dat
        )
      )
      
    }, error = function(e) NULL)
    
    if (!is.null(fit)) {
      
      current_aic <- tryCatch(
        AIC(fit),
        error = function(e) Inf
      )
      
      # NEW: save AIC for this K combination
      aic_table$AIC[r] <- current_aic
      
      if (is.finite(current_aic) &&
          current_aic < best_aic) {
        
        best_aic <- current_aic
        best_model <- fit
        best_K <- current_K
      }
    }
  }
  
  if (is.null(best_model)) {
    stop("All model fits failed.")
  }
  
  beta_estimates <- coef(best_model)
  
  return(list(
    best_model = best_model,
    best_AIC = best_aic,
    best_K = best_K,
    beta = beta_estimates,
    frequency = frequency,
    aic_table = aic_table
  ))
}

tsolr_2 <- function(y, best_K, frequency) {
  
  if(length(best_K) != length(frequency)) {
    stop("best_K and frequency must have same length")
  }
  
  if(!is.ordered(y)) {
    stop("y must be an ordered factor")
  }
  
  n <- length(y)
  t <- seq_len(n)
  response_levels <- levels(y)
  lag_y <- c(NA, as.character(head(y, -1)))
  
  df <- data.frame(y = y)
  
  ref_level <- response_levels[1]
  indicator_names <- character(0)
  
  for(lev in response_levels[-1]) {
    varname <- paste0("lag_", lev)
    df[[varname]] <- as.numeric(lag_y == lev)
    indicator_names <- c(indicator_names, varname)
  }
  
  fourier_names <- character(0)
  num_seas <- length(frequency)
  
  for(s in seq_len(num_seas)) {
    P <- frequency[s]
    K <- best_K[s]
    
    for(k in seq_len(K)) {
      
      cos_name <- paste0("cos_", s, "_", k)
      sin_name <- paste0("sin_", s, "_", k)
      df[[cos_name]] <- cos(2*pi*k*t/P)
      df[[sin_name]] <- sin(2*pi*k*t/P)
      fourier_names <- c(fourier_names, cos_name, sin_name)
    }
  }
  df <- df[-1, ]
  
  rhs_terms <- c(indicator_names, fourier_names)
  
  model_formula <- as.formula(
    paste(
      "y ~",
      paste(rhs_terms, collapse = " + ")
    )
  )
  
  parallel_formula <- as.formula(
    paste(
      "~",
      paste0(paste(fourier_names, collapse = " + "), " - 1")
    )
  )
  
  model <- VGAM::vglm(
    formula = model_formula,
    family = VGAM::cumulative(
      link = "logitlink",
      parallel = parallel_formula,
      reverse = FALSE
    ),
    data = df
  )
  
  model@misc$response_levels <- response_levels
  model@misc$best_K <- best_K
  model@misc$frequency <- frequency
  
  return(model)
}

predict_tsolr <- function(model, train_y, test_y, best_K, frequency) {
  
  if (!is.ordered(train_y)) {
    stop("train_y must be an ordered factor.")
  }
  
  if (length(best_K) != length(frequency)) {
    stop("best_K and frequency must have same length.")
  }
  
  response_levels <- levels(train_y)
  test_y <- factor(test_y, levels = response_levels, ordered = TRUE)
  n_train <- length(train_y)
  n_test <- length(test_y)
  
  if (n_test < 2) {
    stop("Need at least two test observations.")
  }
  
  lag_values <- as.character(test_y[-n_test])
  df_future <- data.frame(dummy = seq_len(n_test - 1))
  
  for (lev in response_levels[-1]) {
    
    varname <- paste0("lag_", lev)
    df_future[[varname]] <- as.numeric(
      lag_values == lev
    )
  }
  
  t_future <- (n_train + 2):(n_train + n_test)
  num_seas <- length(frequency)
  
  for (s in seq_len(num_seas)) {
    
    P <- frequency[s]
    K <- best_K[s]
    
    for (k in seq_len(K)) {
      df_future[[paste0("cos_", s, "_", k)]] <- cos(2 * pi * k * t_future / P)
      df_future[[paste0("sin_", s, "_", k)]] <- sin(2 * pi * k * t_future / P)
    }
  }
  df_future$dummy <- NULL
  
  pred_prob <- predict(
    model,
    newdata = df_future,
    type = "response"
  )
  pred_class <- colnames(pred_prob)[
    max.col(pred_prob, ties.method = "first")
  ]
  
  prob_out <- vector("list", n_test)
  prob_out[[1]] <- NA
  for (i in 2:n_test) {
    prob_out[[i]] <- pred_prob[i - 1, ]
  }
  
  class_out <- rep(NA, n_test)
  class_out[2:n_test] <- as.character(pred_class)
  
  return(list(
    probabilities = prob_out,
    predicted_class = as.numeric(class_out)
  ))
}

tsolr_data <- function(y, best_K, frequency) {
  
  if(length(best_K) != length(frequency)) {
    stop("best_K and frequency must have same length")
  }
  
  if(!is.ordered(y)) {
    stop("y must be an ordered factor")
  }
  
  n <- length(y)
  t <- seq_len(n)
  response_levels <- levels(y)
  lag_y <- c(NA, as.character(head(y, -1)))
  
  df <- data.frame(y = y)
  
  ref_level <- response_levels[1]
  indicator_names <- character(0)
  
  for(lev in response_levels[-1]) {
    varname <- paste0("lag_", lev)
    df[[varname]] <- as.numeric(lag_y == lev)
    indicator_names <- c(indicator_names, varname)
  }
  
  #!!!!!!!!!!!
  #indicator_names <- character(0)
  #!!!!!!!!!!!
  
  fourier_names <- character(0)
  num_seas <- length(frequency)
  
  for(s in seq_len(num_seas)) {
    P <- frequency[s]
    K <- best_K[s]
    
    for(k in seq_len(K)) {
      
      cos_name <- paste0("cos_", s, "_", k)
      sin_name <- paste0("sin_", s, "_", k)
      df[[cos_name]] <- cos(2*pi*k*t/P)
      df[[sin_name]] <- sin(2*pi*k*t/P)
      fourier_names <- c(fourier_names, cos_name, sin_name)
    }
  }
  df <- df[-1, ]
  
  rhs_terms <- c(indicator_names, fourier_names)
  
  model_formula <- as.formula(
    paste(
      "y ~",
      paste(rhs_terms, collapse = " + ")
    )
  )
  
  parallel_formula <- as.formula(
    paste(
      "~",
      paste0(paste(fourier_names, collapse = " + "), "+", paste(indicator_names, collapse = "+"), " - 1")
    )
  )
  
  model <- VGAM::vglm(
    formula = model_formula,
    family = VGAM::cumulative(
      link = "logitlink",
      parallel = parallel_formula,
      reverse = FALSE
    ),
    data = df
  )
  
  model@misc$response_levels <- response_levels
  model@misc$best_K <- best_K
  model@misc$frequency <- frequency
  
  return(model)
}