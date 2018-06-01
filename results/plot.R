library(dplyr)
library(ggplot2)
library(reshape2)

blur <- read.csv('/home/tpin3694/Documents/python/song_lyrics_scraper/results/blur_oasis_results.csv')
queen <- read.csv('/home/tpin3694/Documents/python/song_lyrics_scraper/results/eminem_queen_results.csv')
blur$dataset <- 'Blur vs. Oasis'
queen$dataset <- 'Queen vs. Eminem'

results <- rbind(blur, queen)

results %>% 
  ggplot(aes(x = noise, y = accuracy, colour = classifier)) +
  geom_point(aes(shape=dataset, size =2)) +
  theme_minimal()

results$average <- (results$accuracy+results$precision+results$recall)/3


results %>% 
  ggplot(aes(x=noise, y=accuracy, colour=classifier)) +
  geom_smooth(aes(lty=dataset), se=FALSE) +
  scale_x_continuous(breaks =seq(0, 0.9, by = 0.1), labels=scales::percent)+
  scale_y_continuous(breaks = seq(0.3, 1, by = 0.1), labels = scales::percent) +
  labs(x = 'Amount of Noise', y = 'Accuracy', colour = 'Classifier', title='Effect of Noise on a Classifier\'s Accuracy', lty='Dataset')+
  geom_hline(yintercept=0.5, linetype="dashed", color = "black")+
  geom_label(aes(0.1, 0.5,label = 'Baseline', vjust = 1), colour ='black')+
  theme_bw()

results %>% 
  ggplot(aes(x=noise, y=precision, colour=classifier)) +
  geom_smooth(aes(lty=dataset), se = FALSE) +
  scale_x_continuous(breaks =seq(0, 0.9, by = 0.1), labels=scales::percent)+
  scale_y_continuous(breaks = seq(0.3, 1, by = 0.1), labels = scales::percent) +
  labs(x = 'Amount of Noise', y = 'Precision', colour = 'Classifier', title='Effect of Noise on a Classifier\'s Precision', lty='Dataset')+
  geom_hline(yintercept=0.5, linetype="dashed", color = "black")+
  geom_label(aes(0.1, 0.5,label = 'Baseline', vjust = 1), colour ='black')+
  theme_bw()

results %>% 
  ggplot(aes(x=noise, y=recall, colour=classifier)) +
  geom_smooth(aes(lty=dataset), se = FALSE) +
  scale_x_continuous(breaks =seq(0, 0.9, by = 0.1), labels=scales::percent)+
  scale_y_continuous(breaks = seq(0.3, 1, by = 0.1), labels = scales::percent) +
  labs(x = 'Amount of Noise', y = 'Recall', colour = 'Classifier', title='Effect of Noise on a Classifier\'s Recall', lty='Dataset')+
  geom_hline(yintercept=0.5, linetype="dashed", color = "black")+
  geom_label(aes(0.1, 0.5,label = 'Baseline', vjust = 1), colour ='black')+
  theme_bw()

results %>% 
  ggplot(aes(x=noise, y=average, colour=classifier)) +
  geom_smooth(aes(lty=dataset), se = FALSE) +
  scale_x_continuous(breaks =seq(0, 0.9, by = 0.1), labels=scales::percent)+
  scale_y_continuous(breaks = seq(0.3, 1, by = 0.1), labels = scales::percent) +
  labs(x = 'Amount of Noise', y = '3-metric average', colour = 'Classifier', title='Effect of Noise on a Classifier\'s Overall Performance', lty='Dataset')+
  geom_hline(yintercept=0.5, linetype="dashed", color = "black")+
  geom_label(aes(0.1, 0.5,label = 'Baseline', vjust = 1), colour ='black')+
  theme_bw()

no_lstm <- results %>% 
  filter(classifier!='lstm')
no_lstm_lm <- lm(accuracy ~ 0 +as.factor(classifier) + as.numeric(noise), data = no_lstm)   
summary(no_lstm_lm)


no_lstm_avg <- results %>% 
  filter(classifier!='lstm')
no_lstm_avg_lm <- lm(accuracy~0 + as.factor(classifier) + as.numeric(noise)+as.factor(dataset), data=no_lstm_avg)
summary(no_lstm_avg_lm)


