library(dplyr)
library(class)
library(FNN)

my.knn <- function(train, test, classes, k, regression = FALSE) {

    # TODO For now we assume that test is a single vector
    scaled.combined <- scale(rbind(train, test))
    scaled.train <- scaled.combined[-nrow(scaled.combined), ]
    scaled.test <- scaled.combined[nrow(scaled.combined), ]

    distances <- c()

    # Calculate euclidian distance from test for each
    # row in train
    for (i in 1:nrow(scaled.train)) {
        d <- dist(rbind(scaled.train[i, ], scaled.test))
        distances <- append(distances, as.numeric(d))
    }

    # Create a new dataframe that includes distance from
    # row to the given test row
    train <- cbind(train, distances = distances, classes)

    if (regression == FALSE) {
        prediction <-
            train %>%
            # Sort ascending on distances
            arrange(distances) %>%
            # Take the top k rows by shortest distance
            slice(1:k) %>%
            # Vote on the classes by count
            count(classes) %>%
            # Sort descending
            arrange(-n) %>%
            # Take the top count
            slice(1)

        return(prediction$classes)
    }

    prediction <-
        train %>%
        # Sort ascending on distances
        arrange(distances) %>%
        # Take the top k rows by shortest distance
        slice(1:k) %>%
        # Take the average of the column passed in in classes
        # TODO Don't know how to use a variable here instead
        # of hard-coding the actual column name
        summarise(mean(classes))

    return(prediction)
}

# Remove non-numeric fields, our predicted class
iris_m <- iris[-5]

# Store the classes separately
classes <- iris[5]

for (i in 1:nrow(iris_m)) {
    #predicted <- as.character(my.knn(iris_m[-i, ], iris_m[i, ], classes[-i, ], 10))
    predicted <- as.character(knn(iris_m[-i, ], iris_m[i, ], classes[-i, ], 10))
    test <- as.character(iris[i, ]$Species)
    if (identical(predicted, test) == FALSE) {
        print(paste(i, predicted, test))
    }
}

i = 28

mtcars_m <- select(mtcars, wt, qsec)
mtcars_lm <- select(mtcars, mpg, wt, qsec)
mtcars_train <- mtcars_m[-i, ]
mtcars_train_lm <- mtcars_lm[-i, ]
mtcars_predict <- mtcars_m[i, ]
classes <- select(mtcars, mpg)
my.knn(mtcars_train, mtcars_predict, classes[-i, ], 5, regression = TRUE)

fit <- lm(mpg ~ wt + qsec, data = mtcars_train_lm)
predict(fit, mtcars_predict)

mtcars[i, ]


knn.reg(mtcars_train, mtcars_predict, classes[-i, ], k = 5)


