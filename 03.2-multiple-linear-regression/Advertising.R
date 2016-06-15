advertising <- read.csv("Advertising.csv")

fit <- lm(Sales ~ TV, data=advertising)
summary(fit)
fit
