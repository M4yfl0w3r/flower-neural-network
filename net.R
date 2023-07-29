#!/usr/bin/Rscript

library(datasets)

data(iris)
x <- iris[  , 1:4]
y <- iris[  , 5]

iris$Species <- as.numeric(iris$Species)

num_inputs <- 4
num_neurons <- ncol(x)

generate_random_weights <- function(num_inputs, num_neurons) {
    weights <- vector('list', num_inputs)

    for (i in 1:num_inputs) {
        weights[[i]] <- rnorm(num_neurons)
    }

    return(weights)
}

generate_random_biases <- function(num_neurons) {
    biases <- rnorm(num_neurons)
    return(biases)
}

weights <- generate_random_weights(num_inputs, num_neurons)
biases <- generate_random_biases(num_neurons)

