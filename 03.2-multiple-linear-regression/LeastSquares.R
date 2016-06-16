
# Compare solving (A^T * A)^-1 * A^T * y

A <- matrix(c(rep(1, nrow(mtcars)), mtcars$wt, mtcars$qsec), nrow = nrow(mtcars), ncol = 3)
y <- matrix(mtcars$mpg, nrow = nrow(mtcars), ncol = 1)
solve(t(A) %*% A) %*% t(A) %*% y

s <- summary(lm(mpg ~ wt + qsec, data = mtcars))

m <- matrix(c(s$coeff[1, 1], s$coeff[2, 1], s$coeff[3, 1]), nrow = 3)
m
