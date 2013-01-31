# short R demo, featuring:
#   "pairs" plot
#   randomForest fitting & prediction
#   randomForest variable importance scores


library(randomForest)

loss <- function(actual, predicted) {mean(actual != predicted)}
printf <- function(...) {cat(sprintf(...))}

df_train <- data.frame(read.csv('pima_train.csv'))
n <- nrow(df_train)
m <- ncol(df_train)
x_train <- df_train[, 1:(m-1)]
y_train <- as.factor(df_train[, m])

demo1 <- function() {
    pairs(x_train, col=y_train)
}

demo2 <- function() {
    model <- randomForest(x_train, y_train, ntree=500, do.trace=TRUE,
        importance=TRUE)
    varImpPlot(model)
    y_train_predicted <- predict(model, x_train)
    printf("loss on training data %.3f\n", loss(y_train, y_train_predicted))
}

demo3 <- function() {
    df_test <- data.frame(read.csv('pima_test.csv'))
    x_test <- df_test[, 1:(m-1)]
    y_test <- as.factor(df_test[, m])

    y_test_predicted <- predict(model, x_test)
    printf("loss on test data %.3f\n", loss(y_test, y_test_predicted))
}
