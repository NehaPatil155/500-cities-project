library(tidyverse)
health_data <- read.csv("C:/Users/nehap/Documents/AIT 580/Big data project/500_Cities__Local_Data_for_Better_Health__2017_release_new.csv",sep = ",")
t.test(health_data$PopulationCount,health_data$Data_Value)