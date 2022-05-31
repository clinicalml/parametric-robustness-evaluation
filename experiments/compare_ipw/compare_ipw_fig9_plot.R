setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(tikzDevice)

use.tikz <- T

df <- read_delim("compare_ipw.csv", delim=",", col_types = cols(n="n", method="f"))

# p1 <- subset(df, shift_strength==1) %>%
#   ggplot(aes(x=n, y=abs(loss-base), color=method, fill=method)) + 
#   stat_summary(geom="line", fun=median, alpha=1, size=1)+
#   stat_summary(geom="ribbon", fun.data=median_hilow, fun.args=list(conf.int=0.95), size=0.1, alpha=0.1, show.legend = F) +
#   scale_x_log10() +
#   facet_wrap(~factor(coef_nonlinear)) +
#   theme_minimal()
# 
# print(p1)
# 
# 
# p2 <- subset(df, (coef_nonlinear == 0) & (n==500)) %>%
#   ggplot(aes(x=method, y=loss-base, fill=method)) + 
#   geom_violin() + 
#   theme_minimal() + 
#   facet_wrap(~as.factor(shift_strength))
# 
# print(p2)
# 
# 
# 
# p3 <- ggplot(df, aes(x=coef_nonlinear, y=loss-base, color=method, fill=method)) + 
#   stat_summary(geom="line", fun=median, alpha=1, size=1)+
#   stat_summary(geom="ribbon", fun.data=median_hilow, fun.args=list(conf.int=0.95), size=0.1, alpha=0.1, show.legend = F) +
#   theme_minimal() + 
#   facet_wrap(~shift_strength)
# 
# print(p3)
# 
# p4 <- subset(df, (coef_nonlinear == 0)) %>%
#   ggplot(aes(x=log(abs(loss-base)), fill=method)) + 
#   geom_histogram(alpha=0.7, position = "identity") +
#   facet_grid(n~shift_strength)
# 
# print(p4)

# Setup tikz
path = "../../../shift_gradients_overleaf_clone/figures/compare_ipw"
if(use.tikz){tikz(file=paste0(path, ".tex"), width = 6, height = 2.5)}

nonlinear.names <- c("0"="Linear", "0.5"="Nonlinear")

p5 <- subset(df, (n==500) & (shift_strength <=1.25)) %>%
  mutate(method=str_replace(method, "cap", "clipped")) %>%
  ggplot(aes(x=shift_strength, y=loss - base, group=method,color = method, fill=method)) +
  stat_summary(geom="line", fun=median, alpha=1, size=1)+
  stat_summary(geom="ribbon", fun.data=median_hilow, fun.args=list(conf.int=0.95), size=0.1, alpha=0.05, show.legend = F) +
  facet_wrap(~coef_nonlinear, labeller = as_labeller(nonlinear.names)) +
  labs(x="Shift Strength", y="Prediction Error", color="Method") + 
  theme_minimal()
print(p5)  
if(use.tikz){
  dev.off()
  print(p5)
  lines <- readLines(con=paste0(path, ".tex"))
  lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
  lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
  writeLines(lines,con=paste0(path, ".tex"))
  ggsave(paste0(path, ".pdf"))
  
}
print(p5)