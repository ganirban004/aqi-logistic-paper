# !!! Run Methods.R file before running codes of this file !!!

##### IMPORT DATASET #####
d=read.csv(file.choose(), header = T)





##### PAR(1) AND MTD(1) MODELS #####
ss=366
y=d$Category
l=length(y);
y.training=y[1:(l-ss)]
y.test=y[(l-(ss-1)):l]

pred_MTD <- mtd_fn(y, y.training, y.test)
pred_PAR <- par_fn(y, y.training, y.test)
compute_metrics(actual = y.test, predicted = pred_MTD)
compute_metrics(actual = y.test, predicted = pred_PAR)





##### ML MODELS #####
dml <- d
dml$Category <- factor(dml$Category, levels = 0:5, ordered = T)



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
                         Lag1 = lag(dml$Category, n = 1),
                         AQI = dml$AQI)
model_data <- na.omit(model_data)
model_data <- model_data %>% 
  mutate(I0 = as.numeric(model_data$Lag1 == 0), I1 = as.numeric(model_data$Lag1 == 1), 
         I2 = as.numeric(model_data$Lag1 == 2), I3 = as.numeric(model_data$Lag1 == 3), 
         I4 = as.numeric(model_data$Lag1 == 4))

# splitting train-test set
n <- nrow(model_data)
train_data <- model_data[1:(n - ss), ]
test_data <- model_data[(n - (ss - 1)):n, ]



### TSOLR ###
best_k <- tsolr_1(y = train_data$Response, num_seas = 2, frequency = c(365, 7))
best_k$aic_table
tsolr_model <- tsolr_data(y = train_data$Response, best_K = c(5, 1), frequency = c(365, 7))
actual_tsolr <- as.numeric(test_data$Response) - 1
pred_tsolr <- predict_tsolr(model = tsolr_model, train_y = train_data$Response, 
                            test_y = test_data$Response, best_K = c(5, 1), 
                            frequency = c(365, 7))$predicted_class
pred_tsolr[1] <- 2

compute_metrics(actual_tsolr, pred_tsolr, levels = 0:5)



### ISOLR ###
isolr_model <- vglm(Response ~ Season + Festivity + Daytype + I0 + I1 + I2 + I3 + I4,
                    family = cumulative(link = "logitlink", parallel = ~ Season + Festivity + 
                                          Daytype + I0 + I1 + I2 + I3 + I4 - 1, reverse = FALSE),
                    data = train_data)
pred_isolr <- predict(isolr_model, newdata = test_data, type = "response")
pred_isolr <- as.numeric(colnames(pred_isolr)[max.col(pred_isolr)])
actual_isolr <- as.numeric(test_data$Response) - 1

compute_metrics(actual_isolr, pred_isolr, levels = 0:5)



### Random Forest Model ###
set.seed(123)
rf_model <- randomForest(Response ~ Season + Festivity + Daytype + Lag1, data = train_data)
actual_rf <- as.numeric(test_data$Response) - 1
pred_rf <- as.numeric(predict(rf_model, test_data)) - 1

compute_metrics(actual_rf, pred_rf, levels = 0:5)



### Support Vector Machine ###
svm_model <- svm(Response ~ Season + Festivity + Daytype + Lag1, data = train_data, type = 'C')
actual_svm <- as.numeric(test_data$Response) - 1
pred_svm <- as.numeric(predict(svm_model, test_data)) - 1

compute_metrics(actual_svm, pred_svm, levels = 0:5)



### GETTING PREDICTED VALUES FOR LSTM ###
lstm_pred <- read.csv(file.choose(), header = TRUE)  #choose the data downloaded from LSTM GColab
pred_lstm <- lstm_pred$Model_1



### GETTING PREDICTED VALUES FOR TFT ###
tft_pred <- read.csv(file.choose(), header = TRUE)  #choose the data downloaded from TFT GColab
pred_tft <- tft_pred$Predicted_Value
pred_tft <- c(NA, pred_tft)



### SARIMA ###
sarima_model <- auto.arima(train_data$AQI, seasonal = TRUE)
pred_sarima <- as.numeric(forecast(sarima_model, h = ss)$mean)
pred_sarima <- case_when(
  pred_sarima <= 50 ~ 0, 
  pred_sarima > 50 & pred_sarima <= 100 ~ 1, 
  pred_sarima > 100 & pred_sarima <= 200 ~ 2, 
  pred_sarima > 200 & pred_sarima <= 300 ~ 3, 
  pred_sarima > 300 & pred_sarima <= 400 ~ 4, 
  pred_sarima > 400 ~ 5
)
actual_sarima <- as.numeric(test_data$Response) - 1

compute_metrics(actual_sarima, pred_sarima, levels = 0:5)



### TBATS ###
tbats_model <- tbats(train_data$AQI, seasonal.periods = c(365, 7), use.damped.trend = F)
pred_tbats <- as.numeric(forecast(tbats_model, h = ss)$mean)
pred_tbats <- case_when(
  pred_tbats <= 50 ~ 0, 
  pred_tbats > 50 & pred_tbats <= 100 ~ 1, 
  pred_tbats > 100 & pred_tbats <= 200 ~ 2, 
  pred_tbats > 200 & pred_tbats <= 300 ~ 3, 
  pred_tbats > 300 & pred_tbats <= 400 ~ 4, 
  pred_tbats> 400 ~ 5
)
actual_tbats <- as.numeric(test_data$Response) - 1

compute_metrics(actual_tbats, pred_tbats, levels = 0:5)



### PLOTTING ###
plot_data <- data.frame("Time" = c(1:ss), "Actual" = as.numeric(test_data$Response) - 1, 
                        "ISOLR" = pred_isolr, "TSOLR" = pred_tsolr, 
                        "MTD" = pred_MTD, "PAR" = pred_PAR, 
                        "RF" = pred_rf, "SVM" = pred_svm, 
                        "LSTM" = pred_lstm, "TFT" = pred_tft, 
                        "SARIMA" = pred_sarima, "TBATS" = pred_tbats)

plot_log <- ggplot(plot_data, aes(x = Time)) + 
  geom_line(aes(y = Actual, colour = "Actual")) + geom_line(aes(y = ISOLR, colour = "ISOLR")) + 
  geom_line(aes(y = TSOLR, colour = "TSOLR")) + coord_cartesian(ylim = c(0, 5)) + 
  scale_colour_manual(values = c("Actual" = "black", "ISOLR"  = "#1b9e77", "TSOLR"  = "#d95f02")) + 
  labs(title = "ISOLR & TSOLR", x = "Time Point", y = "AQI Category", color = "Series") + 
  theme_minimal() + 
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 1), legend.position = "bottom")

plot_markov <- ggplot(plot_data, aes(x = Time)) + 
  geom_line(aes(y = Actual, colour = "Actual")) + geom_line(aes(y = MTD, colour = "MTD")) + 
  geom_line(aes(y = PAR, colour = "PAR")) + coord_cartesian(ylim = c(0, 5)) + 
  scale_colour_manual(values = c("Actual" = "black", "MTD"  = "#7570b3", "PAR"  = "#e7298a")) + 
  labs(title = "Markov Models", x = "Time Point", y = "AQI Category", color = "Series") + 
  theme_minimal() + 
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 1), legend.position = "bottom")

plot_ml <- ggplot(plot_data, aes(x = Time)) + 
  geom_line(aes(y = Actual, colour = "Actual")) + geom_line(aes(y = RF, colour = "RF")) + 
  geom_line(aes(y = SVM, colour = "SVM")) + coord_cartesian(ylim = c(0, 5)) + 
  scale_colour_manual(values = c("Actual" = "black", "RF"  = "#66a61e", "SVM"  = "#e6ab02")) + 
  labs(title = "RF & SVM", x = "Time Point", y = "AQI Category", color = "Series") + 
  theme_minimal() + 
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 1), legend.position = "bottom")

plot_dl <- ggplot(plot_data, aes(x = Time)) + 
  geom_line(aes(y = Actual, colour = "Actual")) + geom_line(aes(y = LSTM, colour = "LSTM")) + 
  geom_line(aes(y = TFT, colour = "TFT")) + coord_cartesian(ylim = c(0, 5)) + 
  scale_colour_manual(values = c("Actual" = "black", "LSTM"  = "#a6761d", "TFT"  = "#666666")) + 
  labs(title = "LSTM & TFT", x = "Time Point", y = "AQI Category", color = "Series") + 
  theme_minimal() + 
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 1), legend.position = "bottom")

plot_con <- ggplot(plot_data, aes(x = Time)) + 
  geom_line(aes(y = Actual, colour = "Actual")) + geom_line(aes(y = SARIMA, colour = "SARIMA")) + 
  geom_line(aes(y = TBATS, colour = "TBATS")) + coord_cartesian(ylim = c(0, 5)) + 
  scale_colour_manual(values = c("Actual" = "black", "SARIMA"  = "#a6761d", "TBATS"  = "#666666")) + 
  labs(title = "SARIMA & TBATS", x = "Time Point", y = "AQI Category", color = "Series") + 
  theme_minimal() + 
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 1), legend.position = "bottom")

wrap_plots(plot_log, plot_markov, plot_ml, plot_dl, plot_con, ncol = 2)
