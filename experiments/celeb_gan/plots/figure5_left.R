setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(tikzDevice)
options(tikzLatexPackages 
        =c(getOption( "tikzLatexPackages" ),"\\usepackage{amsfonts}"))

use.tikz = T

df <- read_csv("../plots/random_acc31.csv")
ours.df <- read_csv("../compare_ipw_taylor_optim/results_with_ground_truth_first.csv")
ours <- ours.df$`E_taylor actual`[1]

training <- ours.df$`Training acc`


df$Worse = df$Loss < ours


x.breaks =seq(0.82, 0.96, length.out = 8)
x.labels = paste0("\\scriptsize{$",100*x.breaks,".0\\%$}")

# Setup tikz
path = "../../../../shift_gradients_overleaf_clone/figures/celeba_histogram_31"
if(use.tikz){tikz(file=paste0(path, ".tex"), width = 3.5, height = 1.2)}

p <- ggplot(df, aes(x=Loss, fill=Worse)) + 
  geom_histogram(aes(y=..count../sum(..count..)), bins=14) + 
  geom_vline(aes(colour = "ours", xintercept=ours), linetype="66") + 
  geom_vline(aes(colour = "training", xintercept=training), linetype="66") +
  labs(y=NULL, fill = "\\scriptsize{Random shift acc.}", x="\\scriptsize{Shift distribution acc.}") +
  scale_fill_manual(labels=c("\\scriptsize{Higher than $\\mathbb{E}_\\delta[\\ell]$}", "\\scriptsize{Lower than $\\mathbb{E}_\\delta[\\ell]$}"), values=c("#0F8C2E", "#FDA544")) +
  scale_x_continuous(breaks=x.breaks,labels=x.labels) +
  scale_color_manual(name = NULL, values = c(ours = "red", training = "black"), 
                     labels=c(ours="\\scriptsize{Acc. at $\\delta_{\\texttt{worst-case}}$}",
                              training="\\scriptsize{Training acc.}")) +
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