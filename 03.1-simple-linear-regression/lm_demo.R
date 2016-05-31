library(ggplot2)
library(GGally)

?mtcars

mtcars

str(mtcars)
summary(mtcars)

ggpairs(mtcars[,1:6])

ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point()

ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_smooth(method = "lm", se=FALSE, color="black", formula = y ~ x) +
  geom_point()

cor(mtcars$wt, mtcars$mpg)

fit <- lm(mpg ~ wt, data = mtcars)

fit

summary(fit)

confint(fit)

coef(fit)

str(fit)
