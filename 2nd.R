library("xlsx")
library(foreign)
library(TSA)
library(forecast)
#load data & view series

bush = read.xlsx("F:/Data2/LA/Ass2/Team8.xlsx" , 2)
names(bush)
print(bush)
## convert into date format
bush$Date <- as.Date(bush$Date, "%Y-%m-%d")
str(bush$Date)
range(bush$Date)
## order BY dates
bush <- bush[order(bush$Date), ]

## plot time series
plot(bush$Date, bush$X8802245, type = "l")
head(bush,20)

plot(bush$Date , bush$X8802245)
trend <- lm(X8802245 ~ index(Date), data = bush)
abline(trend, col="red")

detrended.trajectory <- trend$residuals
plot(detrended.trajectory, type="l", main="detrended time series")

busha=bush
##################################################################################
bush$X8802245[is.na(bush$X8802245)] = 0
bush$X9005011[is.na(bush$X9005011)] = 0
bush$X9880099[is.na(bush$X9880099)] = 0
bush$X3244710[is.na(bush$X3244710)] = 0
bush$X6260237[is.na(bush$X6260237)] = 0

busho = bush[c(1,2)]
  
#kingstimeseries <- ts(busho[,2], frequency =365, start=c(2012,10,29))
kingstimeseries <- ts(busho[,2], frequency =100, start=c(2012,10,29))
kingstimeseries
plot(stl(log(kingstimeseries),s.window ="periodic"))

#### USING ZOOO as above method doesnt work ###
Z1.index <- busho[,2]
Z1 = zoo(busho[,2],busho[,1])
Z1 <- zoo(Z1, frequency = 5)
na.approx(Z1)
na.locf(Z1)
na.spline(Z1)
class(Z1)
Z2 = as.ts(Z1)

plot(stl(log(Z2),s.window ="periodic"))

#### Again ZoO
seriezoo=zooreg(busho[,2], frequency=5, start=as.Date("2012-10-29")) 
seriezoo = na.approx(seriezoo)
seriezoo = na.locf(seriezoo)
seriezoo = na.spline(seriezoo)
seriezooTS=as.ts(seriezoo) 
str(seriezooTS) 

decompose(seriezooTS, "mult") 

a = stl((seriezooTS),s.window ="periodic" , na.action = na.fail, robust = TRUE)
plot(stl((seriezooTS),s.window ="periodic" , na.action = na.fail, robust = TRUE))

p1.arima <- auto.arima(seriezooTS)

fit <- HoltWinters(p1.arima,gamma=FALSE)
plot(forecast(fit,h=1))


###
plot(y=bush$X8802245, x=kingstimeseries$Date, type='l')

#identify arima process
acf(bush$X8802245 , na.action = na.pass)
pacf(bush$X8802245 , na.action = na.pass)

#estimate arima model
mod.1 <- arima(bush$X8802245, order=c(0,1,0))
mod.1

#diagnose arima model
acf(mod.1$residuals, na.action = na.pass)
pacf(mod.1$residuals, na.action = na.pass)
Box.test(mod.1$residuals)


fore <- predict(mod.1 , n.ahead=5)
U <- fore$pred + 2*fore$se
L <- fore$pred - 2*fore$se

plot(forecast(mod.1,h=5))

fit <- HoltWinters(mod.1,gamma=FALSE)
plot(forecast(fit,h=5))


plot(bush$X8802245)
lines(seasadj(decompose(bush$X8802245,"multiplicative")),col=4)


tsoutliers(mod.1)
components <- tbats.components(fit)
plot(components)f

croston(bush$X8802245, h=5, alpha=0.1)
cros


