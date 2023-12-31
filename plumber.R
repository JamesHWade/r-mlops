# Generated by the vetiver package; edit with care

library(pins)
library(plumber)
library(rapidoc)
library(vetiver)

# Packages needed to generate model predictions
if (FALSE) {
    library(parsnip)
    library(recipes)
    library(stats)
    library(workflows)
}
b <- board_folder(path = "pins-r")
v <- vetiver_pin_read(b, "penguins_model", version = "20230730T235625Z-54641")

#* @plumber
function(pr) {
    pr %>% vetiver_api(v)
}
