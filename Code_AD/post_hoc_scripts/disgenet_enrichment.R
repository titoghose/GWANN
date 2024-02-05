################################################################################
# DISEASE ENRICHMENT ANALYSIS

library(devtools)
# install_bitbucket("ibi_group/disgenet2r")
library(disgenet2r)
library(ggplot2)
library(stringr)
library(dplyr)
setwd("/home/upamanyu/GWANN/Code_AD/results_Sens8_v4")

disgenet_api_key <- "fc4b82e235b534f3839b7591bb59c6e649af9d66"
Sys.setenv(DISGENET_API_KEY=disgenet_api_key)
# ---------------------------------------------------------------------------- #

nnHits = read.csv("top_100_genes.csv", sep=",", header=T)
res_enrich <- disease_enrichment(entities = nnHits$Gene, vocabulary = "HGNC", database = "ALL")
res_enrich@qresult$Description <- toupper(res_enrich@qresult$Description)
write.table(res_enrich@qresult, file = "enrichments/disgenet_enrichment.tsv", sep = "\t", row.names = FALSE)

res_enrich@qresult$BgRatio_eval <- sapply(res_enrich@qresult$BgRatio, function(x) eval(parse(text=x)))
res_enrich@qresult$Ratio_eval <- sapply(res_enrich@qresult$Ratio, function(x) eval(parse(text=x)))
res_enrich@qresult <- res_enrich@qresult %>% 
                        filter(FDR < 0.05, Count > 5, BgRatio_eval < (5000/21666)) %>%
                        arrange(desc(Ratio_eval)) %>%
                        slice(1:20)

resPlot <- disgenet2r::plot(res_enrich, class = "Enrichment", limit=20, count=5, cutoff= 0.05, nchars=60)
resPlot_out <- resPlot + 
                theme(
                    axis.title.x = element_text(face="plain", size=16),
                    axis.title.y = element_text(face="plain", size=12),
                    axis.text = element_text(face="plain", size=12),
                    title = element_text(face="plain", size=16),
                    legend.title = element_text(face="plain", size=12),
                    legend.text = element_text(face="plain", size=12)) +
                scale_y_discrete(labels = function(x) str_wrap(x, width = 30))

ggsave(filename = "enrichments/disgenet_enrichment.png", dpi = 200, plot = resPlot_out)
ggsave(filename = "enrichments/disgenet_enrichment.svg", plot = resPlot_out)

# ---------------------------------------------------------------------------- #