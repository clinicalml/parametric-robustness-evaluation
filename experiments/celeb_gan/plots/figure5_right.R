library(tidyverse)
library(tikzDevice)
options(tikzLatexPackages 
        =c(getOption( "tikzLatexPackages" ),"\\usepackage{amsfonts}"))
use.tikz = T

df <- read_csv("experiments/celeb_gan/compare_ipw_taylor_optim/results_with_ground_truth.csv")
df$WorstTaylorMinusWorstIPW = df$`E_taylor actual` - df$`E_ipw actual`
df$Worse = df$WorstTaylorMinusWorstIPW > 0


x.breaks =seq(-0.03, 0.01, length.out = 5)
x.labels = paste0("\\scriptsize{$",100*x.breaks,".0\\%$}")

# Setup tikz
path = "experiments/celeb_gan/latex/figures/figure5_right"
if(use.tikz){tikz(file=paste0(path, ".tex"), width = 3.5, height = 1.2)}
p <- ggplot(df, aes(x=df$WorstTaylorMinusWorstIPW, fill=Worse)) + 
  geom_histogram(aes(y=..count../sum(..count..)), breaks = seq(-0.03, 0.01, length.out=13)) +
  labs(y=NULL, x="\\scriptsize{Difference in Shifted acc. ($\\mathbb{E}_{\\delta_{\\texttt{Taylor}}} - \\mathbb{E}_{\\delta_{\\texttt{IS}}}$)}", fill="\\scriptsize{Lower Acc.}") +
  # labs(y=NULL, x="\\scriptsize{Difference in Shifted acc.}", fill="\\scriptsize{Lower Acc.}") +
  scale_fill_manual(labels=c("\\scriptsize{Taylor}", "\\scriptsize{IS}"), values=c("#0F8C2E", "#FDA544")) +
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
