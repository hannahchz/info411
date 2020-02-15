setwd('/Users/hannahchz/Desktop/Project2018_9')
library(randomForest)
library(mlr)
library(caret) #this package has the createDataPartition function
library(forcats)

#read in aircon dataset 
data<-read.csv("new.csv")

data$Brand<-fct_lump(data$Brand, 17)
data$Country<-fct_lump(data$Country, 12)
data$ExpDate<-format(as.Date(data$ExpDate, format="%d/%m/%Y"),"%Y") #0020 
data$ExpDate<-as.factor(data$ExpDate)
cols <- sapply(data, is.logical)
data[,cols] <- lapply(data[,cols], as.numeric)
imp <- impute(data, classes = list(factor = imputeMode(), logical= imputeMode(), integer = imputeMedian(), numeric= imputeMedian()), dummy.classes = c("factor", "logical","integer","numeric"), dummy.type = "numeric")
impdata <- imp$data
new_impdata <- impdata[, which(names(impdata) %in% c("N.Standard", "Configuration2.unitmount", "C.Total.Cool.Rated", "EERtestAvg", "COPtestAvg", "AnnualOutputEER", "AnnualOutputCOP", "Product Class", "outdoortype","sri2010_heat", "Star2010_Cool", "Star2010_Heat", "EER", "Rated.cooling.power.input.kW", "Rated.ACOP", "sri2010_cool"))]

#remove outliers extremely large values
cd <- capLargeValues(new_impdata, target = "sri2010_cool",cols = c("C.Total.Cool.Rated"),threshold = 2.4)
cd <- capLargeValues(cd, target = "sri2010_cool",cols = c("EERtestAvg"),threshold =2.9688)
cd <- capLargeValues(cd, target = "sri2010_cool",cols = c("AnnualOutputEER"),threshold = 2.91183479)
cd <- capLargeValues(cd, target = "sri2010_cool",cols = c("EER"),threshold = 2.9688)
cd <- capLargeValues(cd, target = "sri2010_cool",cols = c("Rated.cooling.power.input.kW"),threshold = 0.4)
cd$C.Total.Cool.Rated<- scale(cd$C.Total.Cool.Rated, scale = FALSE)
cd$EERtestAvg<- scale(cd$EERtestAvg, scale = FALSE)
cd$AnnualOutputEER<- scale(cd$AnnualOutputEER, scale = FALSE)
cd$EER<- scale(cd$EER, scale = FALSE)
cd$Rated.cooling.power.input.kW<- scale(cd$Rated.cooling.power.input.kW, scale = FALSE)
cd$sri2010_cool<- scale(cd$sri2010_cool, scale = FALSE)
cd$sri2010_heat<- scale(cd$sri2010_heat, scale = FALSE)

summarizeColumns(cd) 
str(cd) #90 variables

#new_impdata <- cd[, which(names(cd) %in% c("N-Standard", "Configuration2-unitmount", "C.Total.Cool.Rated", "EERtestAvg", "COPtestAvg", "AnnualOutputEER", "AnnualOutputCOP", "Product Class", "outdoortype","sri2010_heat", "Star2010_Cool", "Star2010_Heat", "EER", "Rated cooling power input kW", "Rated.ACOP", "sri2010_cool"))]

fit_rf = randomForest(sri2010_cool~., data=cd)
# Create an importance based on mean decreasing gini
importance(fit_rf)
varImp(fit_rf)

regr.task = makeRegrTask(data = cd, target = "sri2010_cool")
regr.task

set.seed(1234)

# Define a search space for each learner'S parameter
ps_ksvm = makeParamSet(
  makeNumericParam("sigma", lower = -12, upper = 12, trafo = function(x) 2^x)
)

ps_rf = makeParamSet(
  makeIntegerParam("num.trees", lower = 1L, upper = 200L)
)

# Choose a resampling strategy; 5-fold cross validation
rdesc = makeResampleDesc("CV", iters = 5L)

# Choose a performance measure
meas = rmse

# Choose a tuning method
ctrl = makeTuneControlCMAES(budget = 100L)

# Make tuning wrappers
library(ranger)
tuned.ksvm = makeTuneWrapper(learner = "regr.ksvm", resampling = rdesc, measures = meas,
                             par.set = ps_ksvm, control = ctrl, show.info = FALSE)
tuned.rf = makeTuneWrapper(learner = "regr.ranger", resampling = rdesc, measures = meas,
                           par.set = ps_rf, control = ctrl, show.info = FALSE)
# Four learners to be compared
lrns = list(makeLearner("regr.lm"), tuned.ksvm, tuned.rf)

library(cmaes)
# Conduct the benchmark experiment
bmr = benchmark(learners = lrns, tasks = regr.task, resamplings = rdesc, measures = rmse, 
                show.info = FALSE)
getBMRAggrPerformances(bmr)

plotBMRBoxplots(bmr)

#fit.regr = mlr::train(tuned.ksvm, regr.task)
#library(mmpf)
#pd.ci = generatePartialDependenceData(fit.regr, regr.task, "sri2010_cool", fun = median)
#pd.ci

### Regression gradient boosting machine, specify hyperparameters via a list
regr.lrn = makeLearner("regr.gbm", par.vals = list(n.trees =500, interaction.depth = 5))
### Train the learner
mod = mlr::train(regr.lrn, regr.task)
mod
#mod$features
#getLearnerModel(mod)
### Get the number of observations
n = getTaskSize(regr.task)
### Use 1/3 of the observations for training 
train.set = sample(n, size = n/3)
### Train the learner
mod = mlr::train("regr.lm", regr.task, subset = train.set)
mod
#target = getTaskTargets(regr.task) 
#tab = as.numeric(table(target)) 
#w = 1/tab[target]
#mlr::train("regr.lm", task = regr.task, weights = w)
#Assertion on 'weights' failed: Must have length 3561, but has length 2134.


#we fit a gradient boosting machine to every second observation of the aircon data set and make predictions on the remaining data in regr.task.
n = getTaskSize(regr.task) 
train.set = seq(1, n, by = 2)
test.set = seq(2, n, by = 2)
lrn = makeLearner("regr.gbm", n.trees = 100) 
mod = mlr::train(lrn, regr.task, subset = train.set)
task.pred = predict(mod, task = regr.task, subset = test.set)
task.pred
performance(task.pred)