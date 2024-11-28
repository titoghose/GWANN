################################################################################
# DISEASE ENRICHMENT ANALYSIS

library(devtools)
# install_bitbucket("ibi_group/disgenet2r")
library(disgenet2r)
library(ggplot2)
library(stringr)
library(dplyr)

disgenet_api_key <- "fc1f9b81bcd055c9518522ba15bd1d0638f0ccf4"
Sys.setenv(DISGENET_API_KEY=disgenet_api_key)
# ---------------------------------------------------------------------------- #

setwd("/home/upamanyu/GWANN/Code_AD/results_Sens8_v4")
nnHits = read.csv("top_100_genes.csv", sep=",", header=T)
# res_enrich <- disease_enrichment(entities = nnHits$Gene, vocabulary = "HGNC", database = "ALL")

# setwd("/home/upamanyu/GWANN/Code_AD/results_Sens8_v4/trad_GWAS")
# nnHits = read.csv("trad_GWAS_top_100.csv", sep=",", header=T)

# if else to check if the file exists
enr_file_name <- "enrichments/disgenet_enrichment.tsv" 
if (file.exists(enr_file_name)) {
    qresult <- read.table(enr_file_name, sep = "\t", header = TRUE)
    res_enrich <- new("DataGeNET.DGN", 
                        type = "disease-enrichment", 
                        search = "list", 
                        term = as.character(nnHits$Gene), 
                        database = "ALL",
                        qresult = qresult)
} else {
    res_enrich <- disease_enrichment(entities = nnHits$Gene, vocabulary = "HGNC", database = "ALL")
    res_enrich@qresult$Description <- toupper(res_enrich@qresult$Description)
    write.table(res_enrich@qresult, file = enr_file_name, sep = "\t", row.names = FALSE)
}


res_enrich@qresult$BgRatio_eval <- sapply(res_enrich@qresult$BgRatio, function(x) eval(parse(text=x)))
res_enrich@qresult$Ratio_eval <- sapply(res_enrich@qresult$Ratio, function(x) eval(parse(text=x)))
res_enrich@qresult <- res_enrich@qresult %>% 
                        filter(FDR < 0.05, Count > 5, BgRatio_eval < (5000/21666)) %>%
                        arrange(desc(Ratio_eval)) %>%
                        slice(1:20)
res_enrich@qresult$Description <- str_to_title(res_enrich@qresult$Description)

color_breaks <- (max(res_enrich@qresult$FDR[1:10]) - min(res_enrich@qresult$FDR[1:10])) * c(0.25, 0.75)
color_breaks <- color_breaks + min(res_enrich@qresult$FDR[1:10])
color_breaks <- round(as.numeric(color_breaks), 2)

resPlot <- disgenet2r::plot(res_enrich, class = "Enrichment", limit=10, count=5, cutoff= 0.05, nchars=60)
disgenet_plot <- resPlot + 
                theme_bw() +
                ggtitle("DisGeNET") +
                theme(plot.title = element_text(hjust = 0.5, size = 18),
                        plot.margin = margin(0.5, 1, 0.5, 0.5, "cm"),
                        axis.text = element_text(face = "plain", size = 18),
                        axis.title = element_text(face = "plain", size = 18),
                        legend.title = element_text(size = 16),
                        legend.text = element_text(size = 16),
                        legend.direction = "horizontal",
                        legend.box = "horizontal",
                        legend.position = "bottom",
                        legend.key.width = unit(0.7, 'cm')) +
                labs(color = "FDR") +
                scale_y_discrete(labels = function(x) str_wrap(x, width = 30)) +
                scale_color_continuous(breaks = color_breaks, low="blue", high="red", trans="reverse") +
                guides(size = guide_legend(nrow = 2))

save(disgenet_plot, file = "enrichments/disgenet_plot.RData")
ggsave(filename = "enrichments/disgenet_enrichment.png", width = 10, height = 10, plot = disgenet_plot)
ggsave(filename = "enrichments/disgenet_enrichment.svg", plot = disgenet_plot)
# ---------------------------------------------------------------------------- #