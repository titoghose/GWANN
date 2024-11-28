library(ggplot2)
library(gridExtra)
library(stringr)
library(grid)
library(png)
# setwd("/home/upamanyu/GWANN/Code_AD/results_Sens8_v4")
setwd("/home/upamanyu/GWANN/Code_AD/results_Sens8_v4/trad_GWAS")

load("enrichments/fGSEA_plots.RData")
if (file.exists("enrichments/disgenet_plot.RData")) {
    load("enrichments/disgenet_plot.RData")
} else {
    disgenet_plot <- NULL
}

string_img <- readPNG('enrichments/top_100_STRING.png')
string_grob <- rasterGrob(string_img, interpolate = TRUE)

if (!is.null(disgenet_plot)) {
    print("DisGeNET plot exists")
    comb_plot <- grid.arrange(reactome_plot, wiki_plot, kegg_plot, 
                        go_plot, disgenet_plot, string_grob,
                        ncol=2, nrow=3, widths=c(1, 1), 
                        heights=c(1, 1, 1))
} else {
    comb_plot <- grid.arrange(reactome_plot, wiki_plot, 
                        kegg_plot, go_plot, string_grob,
                        ncol=2, nrow=3, widths=c(1, 1), 
                        heights=c(1, 1, 1))
}

ggsave("enrichments/fGSEA_plots.png", plot=comb_plot, 
        width = 20, height = 24, dpi = 300)
ggsave("enrichments/fGSEA_plots.svg", plot=comb_plot, 
        width = 20, height = 24)
