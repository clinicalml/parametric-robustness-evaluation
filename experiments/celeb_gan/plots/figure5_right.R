setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(tikzDevice)
use.tikz = T

df <- read_csv("../compare_ipw_taylor_optim/results_with_ground_truth.csv")
df$WorstTaylorMinusWorstIPW = df$`E_taylor actual` - df$`E_ipw actual`
df$Worse = df$WorstTaylorMinusWorstIPW > 0


x.breaks =seq(-0.03, 0.02, length.out = 6)
x.labels = paste0("\\scriptsize{$",100*x.breaks,".0\\%$}")

# Setup tikz
path = "../../../../shift_gradients_overleaf_clone/figures/celeba_compare_ipw_taylor_optim"
if(use.tikz){tikz(file=paste0(path, ".tex"), width = 3.5, height = 1.2)}
p <- ggplot(df, aes(x=df$WorstTaylorMinusWorstIPW, fill=Worse)) + 
  geom_histogram(aes(y=..count../sum(..count..)), breaks = seq(-0.03, 0.02, length.out=16)) +
  labs(y=NULL, x="\\scriptsize{Acc. difference at worst shift found}", fill="\\scriptsize{Found worst shift}") +
  scale_fill_manual(labels=c("\\scriptsize{Taylor}", "\\scriptsize{Importance sampling}"), values=c("#0F8C2E", "#FDA544")) +
  scale_x_continuous(breaks=x.breaks,labels=x.labels) +
  theme_minimal() +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.spacing.y = unit(0.1, "cm"),
        legend.key.size = unit(0.4,"line")
        )

print(p)
if(use.tikz){
  dev.off()
  print(p)
  lines <- readLines(con=paste0(path, ".tex"))
  lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
  lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
  writeLines(lines,con=paste0(path, ".tex"))
  ggsave(paste0(path, ".pdf"))
  
}