################################################################################
# DISEASE ENRICHMENT ANALYSIS

library(devtools)
# install_bitbucket("ibi_group/disgenet2r")
library(disgenet2r)
library(ggplot2)

disgenet_api_key <- ""
Sys.setenv(DISGENET_API_KEY=disgenet_api_key)
# ---------------------------------------------------------------------------- #

nnHits = read.table("../results_Sens8_00_GS10_v4/hits.txt", header=T)
res_enrich <- disease_enrichment(entities = nnHits$Gene, vocabulary = "HGNC", database = "ALL")
# res_enrich_cur <- disease_enrichment(entities = nnHits$V1, vocabulary = "HGNC", database = "CURATED")

disTbl <- res_enrich@qresult[1:10, c("Description", "FDR", "Ratio",  "BgRatio")]
res_enrich@qresult$Description <- tolower(res_enrich@qresult$Description)
# resPlot <- plot(res_enrich, class = "Enrichment", cutoff= 0.05, nchars=70)
resPlot <- plot(res_enrich, class = "Enrichment", count = 5,  cutoff= 0.05, nchars=70)
resPlot_out <- resPlot + theme(axis.title.x = element_text(face="bold"),
                axis.title.y = element_text(face="bold"),
                axis.text = element_text(face="plain", size=2),
                title = element_text(face="bold", size=8))
ggsave(filename = "../results_Sens8_00_GS10_v4/enrichments/disgenet_enrichment.png", dpi = 300, plot = resPlot_out)
ggsave(filename = "../results_Sens8_00_GS10_v4/enrichments/disgenet_enrichment.png", plot = resPlot_out)
write.table(res_enrich@qresult, file = "../results_Sens8_00_GS10_v4/enrichments/disgenet_enrichment.tsv", sep = "\t", row.names = FALSE)
# ---------------------------------------------------------------------------- #