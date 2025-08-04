##### IMPORT NECESSARY LIBRARIES #####
library(schoolmath)
library(MASS)
library(NlcOptim)
library(matrixcalc)
library(tidyverse)
library(randomForest)
library(e1071)





##### IMPORT DATASET #####
d=read.csv(file.choose(), header = T)





##### PAR(1) AND MTD(1) MODELS #####
# Mode of a conditional distribution

mode.pred = function(pred.dist){
  a=c(0:(length(pred.dist)-1))
  return(min(a[which(pred.dist==max(pred.dist))]))
} 

# PTP(h) measures

ptp = function(test.sample, pred.sample){
  c = test.sample-pred.sample
  return(length(c[c==0])/length(c)*100)
}

y=d$Category
y

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

l=length(y);l
ss=366

y.training=y[1:(l-ss)];length(y.training)
y.test=y[(l-(ss-1)):l];length(y.test)

n_s=length(y.training)

kk=max(y.training)+1;kk

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

# storing the PTPs
round(predictedErrorMTD(y,y.training,y.test),2) # for MTD
round(predictedErrorPAR(y,y.training,y.test),2) # for PAR





##### ML MODELS #####
dml <- d
dml$Category <- factor(dml$Category, levels = 0:5, ordered = T)



### Making Performance Metrics ###
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



### Making Festive Dates ###
dml <- dml %>% mutate(Date_full = make_date(Year, match(Month, month.name), Date))
diwali_dates <- dml %>% filter(Is.Diwali == "Diwali") %>% dplyr::select(Year, Diwali_Date = Date_full)
dml <- dml %>% left_join(diwali_dates, by = "Year") %>% 
  mutate(
    Days_from_Diwali = as.integer(Date_full - Diwali_Date), 
    Festivity = case_when(
      Days_from_Diwali >= -7  & Days_from_Diwali <= 7 ~ "Diwali",
      TRUE ~ "Non-diwali"))
rm(diwali_dates)
dml$Date_full <- NULL
dml$Days_from_Diwali <- NULL
dml$Diwali_Date <- NULL



### Making Seasons ###
dml <- dml %>% mutate(Season = case_when(
  Month %in% c("March", "April", "May") ~ "Summer",
  Month %in% c("June", "July", "August", "September") ~ "Monsoon",
  Month %in% c("October", "November") ~ "Autumn",
  Month %in% c("December", "January", "February") ~ "Winter"))



### Adding Predictor Variables ###
model_data <- data.frame(t = 1:nrow(dml), 
                         Response = dml$Category, 
                         Season = dml$Season, 
                         Festivity = dml$Festivity, 
                         Daytype = dml$Daytype, 
                         Lag1 = lag(dml$Category, n = 1)) %>% mutate(Season_sin = sin(2 * pi * t / 365), 
                                                                     Season_cos = cos(2 * pi * t / 365), 
                                                                     Diwali_sin = abs(sin(1 * pi * t / 365)), 
                                                                     Diwali_cos = abs(cos(1 * pi * t / 365)), 
                                                                     Daytype_sin = sin(2 * pi * t / 7), 
                                                                     Daytype_cos = cos(2 * pi * t / 7))
model_data <- na.omit(model_data)

# splitting train-test set
n <- nrow(model_data)
train_data <- model_data[1:(n - ss), ]
test_data <- model_data[(n - (ss - 1)):n, ]



### Logistic Regression ###
log_model_1 <- polr(Response ~ Season + Festivity + Daytype + Lag1, data = train_data, Hess = TRUE)
log_model_2 <- polr(Response ~ Season_sin + Season_cos + Diwali_sin + Diwali_cos + 
                      Daytype_sin + Daytype_cos + Lag1, data = train_data, Hess = TRUE)

actual_log <- as.numeric(test_data$Response) - 1
pred_1_log <- as.numeric(predict(log_model_1, test_data)) - 1
pred_2_log <- as.numeric(predict(log_model_2, test_data)) - 1

compute_metrics(actual_log, pred_1_log)
compute_metrics(actual_log, pred_2_log)



### Random Forest Model ###
set.seed(123)
rf_model_1 <- randomForest(Response ~ Season + Festivity + Daytype + Lag1, data = train_data)
rf_model_2 <- randomForest(Response ~ Season_sin + Season_cos + Diwali_sin + Diwali_cos + 
                             Daytype_sin + Daytype_cos + Lag1, data = train_data)
actual_rf <- as.numeric(test_data$Response) - 1
pred_1_rf <- as.numeric(predict(rf_model_1, test_data)) - 1
pred_2_rf <- as.numeric(predict(rf_model_2, test_data)) - 1

compute_metrics(actual_rf, pred_1_rf)
compute_metrics(actual_rf, pred_2_rf)



### Support Vector Machine ###
svm_model_1 <- svm(Response ~ Season + Festivity + Daytype + Lag1, data = train_data, type = 'C')
svm_model_2 <- svm(Response ~ Season_sin + Season_cos + Diwali_sin + Diwali_cos + 
                     Daytype_sin + Daytype_cos + Lag1, data = train_data, type = 'C')

actual_svm <- as.numeric(test_data$Response) - 1
pred_1_svm <- as.numeric(predict(svm_model_1, test_data)) - 1
pred_2_svm <- as.numeric(predict(svm_model_2, test_data)) - 1

compute_metrics(actual_svm, pred_1_svm)
compute_metrics(actual_svm, pred_2_svm)
