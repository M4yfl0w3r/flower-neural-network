#!/usr/bin/Rscript

library(datasets)

data(iris)
x <- iris[  , 1:4]
y <- iris[  , 5]

iris$Species <- as.numeric(iris$Species)

print(y)
