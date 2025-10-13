rm(list = ls()) # clean workspace
library(wCorr)
# 获取命令行参数, 6为第一个参数
Args <- commandArgs()
# Rscript comp.r <file name> <meter number/strength>
# read data
merged <- read.delim(Args[6], header = TRUE, quote = "")
# config used data and methods here
LIST_DATA   = c(Args[7])
LIST_METHOD = c("wspearman")
REF    = merged$strength # used as refernce
WEIGHT = merged$weight   # used as weight
###########################
results = matrix(numeric(0),
                 nrow = length(LIST_DATA),
                 ncol = length(LIST_METHOD))
# main loop
row_index = 0
for (COL in LIST_DATA) {
  row_index = row_index + 1
  cat(COL, "\t")
  SAM = merged[, COL]
  col_index = 0
  for (METHOD in LIST_METHOD) {
    col_index = col_index + 1
    if (METHOD == "wspearman") {
      C = weightedCorr(REF, SAM, method = "spearman", weights = WEIGHT)
    } else{
      C = -1
    }
    cat(round(C, digits = 3), "\t") # output to stdout
  }
  cat("\n")
}
# -1.0 means strong negative correlation; meter works but 'strong' passwords are in fact 'weak' and the other way round
#  0.0 means no correlation; meter is randomly guessing, and not estimating password strength
#  1.0 means strong positive correlation; meter works perfectly
