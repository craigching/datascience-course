
# Simple linear regression

fit <- lm(mpg ~ wt, data = mtcars)
summary(fit)

# Multiple linear regression fitting mpg vs all
# other variables

fit <- lm(mpg ~ ., data = mtcars)
summary(fit)

# Multiple linear regressing fitting mpg vs some
# random variables of my choosing

fit <- lm(mpg ~ wt + cyl + disp, data = mtcars)
summary(fit)

# Find the best model

fit <- step(lm(mpg ~ ., data = mtcars))
summary(fit)

# A crude correlation matrix

cor(mtcars)
