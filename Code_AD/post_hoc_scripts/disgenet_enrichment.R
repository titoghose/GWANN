################################################################################
# DISEASE ENRICHMENT ANALYSIS

library(devtools)
# install_bitbucket("ibi_group/disgenet2r")
library(disgenet2r)
library(ggplot2)
setwd("/home/upamanyu/GWANN/Code_AD/results_Sens8_v4")

disgenet_api_key <- "fc4b82e235b534f3839b7591bb59c6e649af9d66"
Sys.setenv(DISGENET_API_KEY=disgenet_api_key)
# ---------------------------------------------------------------------------- #

nnHits = read.table("enrichments/unpruned_gene_hits_1e-14.csv", header=T)
res_enrich <- disease_enrichment(entities = nnHits$Gene, vocabulary = "HGNC", database = "ALL")
res_enrich@qresult$Description <- tolower(res_enrich@qresult$Description)
write.table(res_enrich@qresult, file = "enrichments/disgenet_enrichment.tsv", sep = "\t", row.names = FALSE)
# res_enrich_cur <- disease_enrichment(entities = nnHits$V1, vocabulary = "HGNC", database = "CURATED")

res_enrich@qresult$BgRatio_eval <- sapply(res_enrich@qresult$BgRatio, function(x) eval(parse(text=x)))
res_enrich@qresult <- res_enrich@qresult[res_enrich@qresult$BgRatio_eval < (5000/21666),]
resPlot <- plot(res_enrich, class = "Enrichment", count=5, cutoff= 0.05, nchars=70)
resPlot_out <- resPlot + theme(axis.title.x = element_text(face="bold"),
                axis.title.y = element_text(face="bold"),
                axis.text = element_text(face="plain", size=2),
                title = element_text(face="bold", size=8))

ggsave(filename = "enrichments/disgenet_enrichment.png", dpi = 200, plot = resPlot_out)
ggsave(filename = "enrichments/disgenet_enrichment.svg", plot = resPlot_out)

# ---------------------------------------------------------------------------- #