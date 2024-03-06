library(LDlinkR)

args <- commandArgs(trailingOnly = TRUE)
out_file <- args[1]
snps <- read.table(args[2], header = FALSE)$V1

mat <- LDmatrix(snps, pop = "GBR", r2d = "r2", token = "81b0b1b59294",
        file = out_file, genome_build = "grch37", 
        api_root = "https://ldlink.nih.gov/LDlinkRest")