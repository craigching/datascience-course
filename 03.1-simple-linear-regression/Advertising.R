library(ggplot2)

advertising <- read.csv("Advertising.csv")

fit <- lm(Sales ~ TV, data=advertising)
summary(fit)

ggplot(advertising, aes(x = TV, y = Sales)) +
  geom_smooth(method = "lm", se=FALSE, color="red", formula = y ~ x) +
  geom_point()

