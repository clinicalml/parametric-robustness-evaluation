setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(tikzDevice)
library(reshape2)

use.tikz <- T
for.workshop <- T
df <- read_csv("labtest_variance_R.csv") %>%
  mutate(variable=factor(variable, levels=c("order", "age")))


# Setup tikz
w = ifelse(for.workshop, 3, 6)
h = ifelse(for.workshop, 2.0, 3.5)
path = paste0("../../../../shift_gradients_", ifelse(for.workshop, "workshop", "overleaf"), "_clone/figures/labtest_variance")
if(use.tikz){tikz(file=paste0(path, ".tex"), width = w, height = h)}


var.names <- c("age"="\\scriptsize{Age}", "order"="\\scriptsize{Test ordering}")
method.names <- c("ipw"="\\scriptsize{Importance samp.}", "taylor"="\\scriptsize{Taylor}", "true"="\\scriptsize{Ground truth}")


p <- ggplot(df, aes(x=delta, y = mean, fill=method, color=method, linetype=method)) +
  geom_line(size=1) + 
  geom_ribbon(aes(ymin=mean-2*std,ymax=mean+2*std), alpha=0.05, size=0.25) +
  facet_wrap(~variable, labeller = as_labeller(var.names)) +
  labs(fill="\\scriptsize{Estimate}", color="\\scriptsize{Estimate}", x="\\scriptsize{Shift strength $\\delta$}", y="\\scriptsize{Shift loss}")+
  coord_cartesian(ylim=c(0, 1.2)) +
  scale_fill_brewer(palette="Dark2", breaks=names(method.names), labels=method.names)+
  scale_color_brewer(palette="Dark2", breaks=names(method.names), labels=method.names)+
  theme_minimal() +
  scale_linetype_manual(values=c(1, 1, 8), guide="none")+
  theme(legend.position = "bottom", 
        legend.spacing.y = unit(0.1, "cm"),
        legend.key.size = unit(0.4,"line"),
        legend.margin=margin(0,0,0,0),
        legend.box.margin=margin(-10,-10,0,-10))

print(p)

if(use.tikz){
  dev.off()
  print(p)
  ggsave(paste0(path, ".pdf"))
  
}
print(p)
