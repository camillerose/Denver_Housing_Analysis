# Denver Housing Data Analysis for Insight

# This table is comprised of the transfer of property ownership via sales 
# or through other means with a sales year of 2008 to present.

# Refer to RealPropertyCodeBook for property descriptors

# load relevant libraries
library(ggplot2)
library(GGally)
library(Hmisc)
library(corrplot)
library(PCAmixdata)
library(caret)
library(caTools)
library(relaimpo)
library(dplyr)

# set working directory to file location
setwd("~/Desktop/Data Science Courses/My Mini Projects/Denver Housing Analysis")

# read in data sets 
sales_data <- read.csv("sales.csv", stringsAsFactors = FALSE) #sale data
char_data <- read.csv("additional_characteristics.csv", stringsAsFactors = FALSE) #housing characteristics

####################### *START HERE* #######################

######################## CLEAN DATA ########################

# subset char_data and sales_data for values of interest
sales_sub <- dplyr::select(sales_data, PIN, SALE_YEAR, D_CLASS_CN, SALE_PRICE, NBHD_1_CN)
char_sub<- dplyr::select(char_data, PIN, LAND_SQFT, AREA_ABG, BED_RMS, 
                  FULL_B, HLF_B, BSMT_AREA, FBSMT_SQFT)

# rename variable names that aren’t intuitive
colnames(sales_sub)[3] <- "PROPERTY_TYPE"
colnames(sales_sub)[5] <- "NEIGHBORHOOD"
colnames(char_sub)[3] <- "ABOVE_GRD_SQFT"

# make nice tibble
char <- tbl_df(char_sub)
sales <- tbl_df(sales_sub)

# merge data sets based on PIN match between the two data sets
combined <- merge(sales,char, by = "PIN")
denver <- tbl_df(combined)

# only keep rows where property type is single family
inds <- denver$PROPERTY_TYPE == "SINGLE FAMILY" 
denver <- denver[inds,]
denver <- dplyr::select(denver, -PROPERTY_TYPE, -PIN)

# remove NAs - show
inds <- which(!is.na(denver$SALE_PRICE) & !is.na(denver$LAND_SQFT)
              & !is.na(denver$FULL_B) & !is.na(denver$HLF_B))

denver <- denver[inds,]

# make factors out of some variables for the lm 
denver$NEIGHBORHOOD <- as.factor(denver$NEIGHBORHOOD)
denver$SALE_YEAR <- as.factor(denver$SALE_YEAR)

# neighborhoods of interest: "W BAKER" "BAKER" "LINCOLN PARK" "CAPITOL HILL S" "WHITTER"
inds <- which(denver$NEIGHBORHOOD == "W BAKER" | denver$NEIGHBORHOOD == "BAKER" |
        denver$NEIGHBORHOOD == "LINCOLN PARK" | denver$NEIGHBORHOOD == "CAPITOL HILL S")

denver <- denver[inds,]

# narrow down sale year 2012-2018
inds <- which(denver$SALE_YEAR == "2012" | denver$SALE_YEAR == "2013" | denver$SALE_YEAR == "2014" |
                denver$SALE_YEAR == "2015" | denver$SALE_YEAR == "2016" | denver$SALE_YEAR == "2017" |
                denver$SALE_YEAR == "2018")

denver <- denver[inds,]

summary(denver)

# remove outliers
# some sale price values are too low, unsure of meaning, possible foreclosures
# keep sale price values over 100000 only, under 99th percentile
quant <- quantile(denver$SALE_PRICE, c(0.97))
inds <- which(denver$SALE_PRICE > 100000 & denver$SALE_PRICE < quant[[1]]) 
denver <- denver[inds,]


######################## SUMMARIZE DATA ########################

# inspect summary of data for anything weird
summary(denver)

# check for normality with a histogram
hist(denver$SALE_PRICE)

# The central limit theorem tells us that 
# no matter what distribution things have, the sampling 
# distribution tends to be normal if the sample is large enough.


######################## INSPECT CORRELATIONS ########################

# pair-wise comparisons for collinearity

denver_pair_plot <- dplyr::select(denver, -SALE_YEAR, -NEIGHBORHOOD) # only numerics

cors <- round(cor(denver_pair_plot),2)

# Plot
corrplot::corrplot(cors, method = "number", type = "upper") 

# FBSMT_SQFT is correlated with BMST_AREA (0.77), throw out FBSMT_SQFT 
# because of collinearity 


######################## DO PCA #########################

# do a principle components analysis to see how many 
# variables to include in the model // how many components explain variance

# split the data into qualitative and quantitative variables

# split1 <- splitmix(denver)
# X1 <- split1$X.quanti 
# X2 <- split1$X.quali 
# res.pcamix <- PCAmix(X.quanti=X1, X.quali=X2,rename.level=TRUE,
#                      graph=FALSE)


denver_pca <- prcomp(denver_pair_plot, center = TRUE, scale. = TRUE) 
plot(denver_pca, type = "l",col=3) # from the plot we see 3 +/- 1 variables


######################## BUILD THE MODEL #########################

set.seed(128)
# randomly sample 75% of the number of rows in the denver data for training
split <- sample(seq_len(nrow(denver)), size = floor(0.75 * nrow(denver)), replace = F) 

# select training data
trainData <- denver[split, ]

# select the rest of the data for testing data (the remaining 25%)
testData <- denver[-split, ]

# build the prediction model with training data
predictionModel <- lm(SALE_PRICE ~ SALE_YEAR + NEIGHBORHOOD + LAND_SQFT + 
                      ABOVE_GRD_SQFT + BED_RMS + FULL_B + HLF_B + BSMT_AREA, 
                      data = trainData)
# show summary
summary(predictionModel)

# remove variables - remove BED_RMS
### FINAL MODEL ### Adjusted R-squared of ~0.63
predictionModel <- lm(SALE_PRICE ~ SALE_YEAR + NEIGHBORHOOD + LAND_SQFT + 
                        ABOVE_GRD_SQFT + FULL_B + HLF_B, 
                      data = trainData)

# show summary
summary(predictionModel)

# calculate relative contributuons of different variables to r^2
# lmg is the r^2 contribution averaged over orderings among regressors
cont<-calc.relimp(predictionModel, type = c("lmg"), rela = TRUE)
plot(cont)

######################## TEST THE MODEL #########################

# Test prediction model
predictionTest <- predict(predictionModel, newdata = testData)

# Calculate R-squared for prediction model
SSE <- sum((testData$SALE_PRICE - predictionTest) ^ 2)
SST <- sum((testData$SALE_PRICE - mean(testData$SALE_PRICE)) ^ 2)

# r-squared prediction is 0.64 - not bad, not great
# => 0.64
r_sq_prediction <- (1 - (SSE/SST))

# calculate prediction accuracy and error rates
actuals_preds <- data.frame(cbind(actuals=testData$SALE_PRICE, predicteds=predictionTest))  # make actuals_predicteds dataframe

# correlation accuracy
# => 0.80
correlation_accuracy <- cor(actuals_preds) 

# min/max accuracy: If predict (the column predicteds in your data frame) 
# exactly equals actual (actuals) for every instance of the test set, the 
# row minimum would be the same as the row maximum, so the ratio would be 1.0 for all rows.
# => 0.83, 83%
min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))  

# mean absolute percentage error/deviation (smaller is better)
# => 21%
mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals)  

# RSE
# => .22, 22% error rate
rse <- sigma(predictionModel)/mean(denver$SALE_PRICE)

# plot residuals
# we can see the model was very off sometimes
options(scipen=5)
plot(residuals(predictionModel), ylab = "Residuals", main = "Residuals Plot")
abline(h=0, col="red")

# plot observed vs. predicted --- plot abline step wise
# bad at predicting more expensive house prices
par(xaxs="i", yaxs="i") 
options(scipen=5)
with(actuals_preds,
     plot(actuals, predicteds, ylim=c(0,700000), 
          xlim=c(0,700000),
          main="Observed vs. Predicted",pch=1,
          xlab = "Observed Sale Price", ylab = "Predicted Sale Price", col="blue"))

abline(a=0,b=1,col="red")

######################## USE MODEL FOR HOUSE SEARCH #########################

# Predict value of 444 INCA house

inca444 <- data.frame("SALE_YEAR" = c("2018"), "SALE_PRICE" = c(0), "NEIGHBORHOOD" = c("W BAKER"), "LAND_SQFT" = c(3182), 
           "ABOVE_GRD_SQFT" = c(850), "BED_RMS" = c(2), "FULL_B" = c(1), "HLF_B" = c(0), "BSMT_AREA" = c(400),
           "FMSBT_SQFT" = c(0))

# make factors out of some variables for the lm 
inca$NEIGHBORHOOD <- as.factor(inca$NEIGHBORHOOD)
inca$SALE_YEAR <- as.factor(inca$SALE_YEAR)

# make into tibble for quick glance
inca_tbl <- tbl_df(inca444)

# predict what 444 Inca should cost
predictionInca <- predict(predictionModel, newdata = inca_tbl)

predictionInca
