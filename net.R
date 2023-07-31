#!/usr/bin/Rscript

library(datasets)

data(iris)
x <- iris[4, 1:4]
y <- iris[, 5]

iris$Species <- as.numeric(iris$Species)

NUM_INPUTS  <- 4
NUM_NEURONS <- 3
INPUT_SIZE  <- nrow(x)

generate_random_weights <- function(num_inputs, num_neurons) {
    weights <- vector('list', NUM_INPUTS)

    for (i in 1:NUM_INPUTS) {
        weights[[i]] <- rnorm(NUM_NEURONS)
    }

    return(weights)
}

generate_random_biases <- function(num_neurons) {
    return(rnorm(NUM_NEURONS))
}

dense_layer <- function(input) {
    weights <- generate_random_weights(NUM_INPUTS, NUM_NEURONS)
    biases  <- generate_random_biases(NUM_NEURONS)
    output  <- vector('list', NUM_NEURONS)

    for (i in 1:NUM_NEURONS) {
        output[[i]] <- sum(input * weights[[i]]) + biases[i]
    }

    return(output)
}

output <- dense_layer(input = x)
print(output)
