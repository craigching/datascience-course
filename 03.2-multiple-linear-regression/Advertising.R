# Fit the advertising data to three models and look at
# RSE, R^2, and Ajusted R^2

advertising <- read.csv("Advertising.csv")

fit1 <- summary(lm(Sales ~ TV, data=advertising))
fit2 <- summary(lm(Sales ~ TV + Radio, data=advertising))
fit3 <- summary(lm(Sales ~ TV + Radio + Newspaper, data=advertising))

fit1$r.squared
fit2$r.squared
fit3$r.squared

fit1$sigma
fit2$sigma
fit3$sigma

fit1$adj.r.squared
fit2$adj.r.squared
fit3$adj.r.squared

advertising$X <- NULL
cor(advertising)
