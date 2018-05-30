library(dplyr)
library(ggplot2)
library(reshape2)

results <- read.csv('/home/tpin3694/Documents/python/song_lyrics_scraper/results/noise_results.csv')

results %>% 
  select(c(noise, lo_acc, lo_acc_se, rf_acc, rf_acc_se)) %>% 
  ggplot(aes(x=noise)) +
  geom_line(aes(y=lo_acc), colour='red') +
  geom_line(aes(y=rf_acc), colour='blue')+
  theme_minimal()
