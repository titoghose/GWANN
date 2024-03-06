library(dplyr)
library(ggplot2)
library(stringr)
library(fgsea)
library(biomaRt)
library(gridExtra)

replaceBioTerms <- function(terms) {
    terms <- gsub("Gtpase", "GTPase", terms)
    terms <- gsub("Rho", "RHO", terms)
    terms <- gsub("Rac1", "RAC1", terms)
    terms <- gsub("Egf", "EGF", terms)
    terms <- gsub("EGFr", "EGFR", terms)
    terms <- gsub("Pi3k", "PI3K", terms)
    terms <- gsub("Akt", "AKT", terms)
    terms <- gsub("Vegfa Vegfr2", "VEGFA VEGFR2", terms)
    terms <- gsub("Abc", "ABC", terms)
    terms <- gsub("Bdnf", "BDNF", terms)
    terms <- gsub("Erbb", "ERBB", terms)
    terms <- gsub("\\(Cc\\)", "\\(CC\\)", terms)
    terms <- gsub("\\(Bp\\)", "\\(BP\\)", terms)
    terms <- gsub("\\(Mf\\)", "\\(MF\\)", terms)
    return(terms)
}

GSEA = function(gene_list, pval, top_100_genes, hits,
                pathway_set = 'Reactome', pathway_file = NULL, 
                maxsize = 500) {
    if ( any( duplicated(names(gene_list)) )  ) {
        warning("Duplicates in gene names")
        gene_list = gene_list[!duplicated(names(gene_list))]
    }
    if  ( !all( order(gene_list, decreasing = TRUE) == 1:length(gene_list)) ){
        warning("Gene list not sorted")
        gene_list = sort(gene_list, decreasing = TRUE)
    }
    
    if(is.null(pathway_file)) {
        pws <- reactomePathways(names(gene_list))
    } else {
        pws <- gmtPathways(pathway_file)
    }
    
    # GSEA
    set.seed(54321)
    fgRes <- fgsea::fgseaMultilevel(pathways=pws, 
                            stats=gene_list,
                            maxSize=maxsize,
                            scoreType="pos") %>% 
                    as.data.frame() %>% 
                    dplyr::filter(padj < !!pval)
    fgRes <- as.data.frame(fgRes[order(fgRes$pval), ])
    fgRes$leadingEdge <- sapply(fgRes$leadingEdge, paste, collapse=";")
    if(nrow(fgRes) == 0) {
        return()
    }
    # Find list intersection of leading edge with hits and convert to string
    # seperated by ;
    fgRes$hits_overlap <- sapply(fgRes$leadingEdge, 
                                    function(x) {
                                        paste(intersect(unlist(strsplit(x, ";")), hits), collapse=";")
                                    })
    fgRes$top_100_overlap <- sapply(fgRes$leadingEdge, 
                                    function(x) {
                                        length(intersect(unlist(strsplit(x, ";")), top_100_genes)) 
                                    })
    fgRes <- fgRes  %>% mutate(top_100_ratio = top_100_overlap / length(top_100_genes))

    write.csv(fgRes, file = paste("enrichments/fGSEA_", pathway_set, ".csv", sep=""), 
                quote=TRUE, 
                row.names=FALSE)
    
    plot_df <- fgRes %>% filter(top_100_overlap > 0)
    plot_df$pathway <- gsub("REACTOME_", "", plot_df$pathway)
    plot_df$pathway <- gsub("KEGG_", "", plot_df$pathway)
    plot_df$pathway <- gsub("WP_", "", plot_df$pathway)
    plot_df$pathway <- gsub("GOCC_", "(CC)_", plot_df$pathway)
    plot_df$pathway <- gsub("GOBP_", "(BP)_", plot_df$pathway)
    plot_df$pathway <- gsub("GOMF_", "(MF)_", plot_df$pathway)
    plot_df$pathway <- gsub("_", " ", plot_df$pathway)
    plot_df$pathway <- str_to_title(plot_df$pathway)
    plot_df$pathway <- replaceBioTerms(plot_df$pathway)
    
    # plot_df$overlap <- sapply(plot_df$hits_overlap, function(x) length(unlist(strsplit(x, ";"))))
    # plot_df <- plot_df[plot_df$overlap != 0,]
    plot_df <- plot_df[order(plot_df$NES, decreasing = TRUE),]
    plot_df <- head(plot_df, 10)
    plot_df <- plot_df[order(plot_df$NES),]

    plot_df$pathway <- factor(plot_df$pathway, levels = plot_df$pathway)
    plot_df$padj <- -log10(plot_df$padj)

    color_breaks <- (max(plot_df$padj) - min(plot_df$padj)) * c(0.25, 0.75)
    color_breaks <- color_breaks + min(plot_df$padj)
    color_breaks <- round(as.numeric(color_breaks), 2)
    
    plot <- ggplot(plot_df, aes(x = NES, y = pathway, color = padj)) +
                geom_point(size = 6) +
                scale_color_continuous(breaks=color_breaks, low = "blue", high = "red") +
                theme_bw() +
                labs(x = "NES", 
                    y = "",
                    color = "-log10(P-value)") +
                ggtitle(paste(pathway_set)) +
                theme(plot.title = element_text(hjust = 0.5, size = 18),
                        plot.margin = margin(0.5, 0.5, 0.5, 0.4, "cm"),
                        axis.text = element_text(face = "plain", size = 18),
                        axis.title = element_text(face = "plain", size = 18),
                        legend.title = element_text(size = 16),
                        legend.text = element_text(size = 16),
                        legend.direction = "horizontal",
                        legend.box = "vertical",
                        legend.position = "bottom",
                        legend.key.width = unit(0.9, 'cm')) +
                scale_y_discrete(labels = function(x) str_wrap(x, width = 30))
    ggsave(paste("enrichments/fGSEA_", pathway_set, ".png", sep=""), 
            plot, width = 10, height = 10)
    return (plot)
}

# GWANN
setwd("/home/upamanyu/GWANN/Code_AD/results_Sens8_v4")
stats <- read.csv("summary_nsuperwins_1_gene_level.csv", sep=",")
glist <- as.vector(1 - stats$stat_trial_A)
top_100_genes <- read.csv("top_100_genes.csv")$Gene
hits <- read.csv("LD/r20.8_pruned_gene_hits_1e-25.csv") %>% filter(pruned == "False")
hits <- hits$Gene
names(glist) <- stats$Gene

# Traditional GWAS
# setwd("/home/upamanyu/GWANN/Code_AD/results_Sens8_v4/trad_GWAS")
# stats <- read.csv("trad_GWAS_summary.csv", sep=",")
# glist <- as.vector(abs(stats$Z_STAT))
# names(glist) <- stats$symbol
# top_100_genes <- read.csv("trad_GWAS_top_100.csv")
# top_100_genes <- top_100_genes$symbol
# hits <- read.csv("LD/pruned_gene_hits_5e-08.csv") %>% filter(pruned == "False")
# hits <- hits$symbol

# Get entrez gene ids from hgnc symbols
if (file.exists("entrez_glist.csv")) {
    entrez_glist <- read.csv("entrez_glist.csv")
    entrez_glist <- entrez_glist %>% mutate_all(as.character)
} else {
    mart <- useEnsembl(biomart="ensembl", dataset="hsapiens_gene_ensembl", 
                    mirror = "asia")
    entrez_glist <- getBM(attributes=c("hgnc_symbol", "entrezgene_id"), 
                filters="hgnc_symbol", 
                values=names(glist), 
                mart=mart)
    write.csv(entrez_glist, file = "entrez_glist.csv", row.names=FALSE, quote=TRUE)
}
entrez_glist <- entrez_glist[!duplicated(entrez_glist$entrezgene_id), ]

glist_m <- merge(glist, entrez_glist, by.x="row.names", by.y="hgnc_symbol")
glist_m <- glist_m[order(glist_m$x, decreasing = TRUE), ]
glist <- glist_m$x
names(glist) <- glist_m$entrezgene_id

top_100_genes <- entrez_glist[entrez_glist$hgnc_symbol %in% top_100_genes, "entrezgene_id"]
hits <- entrez_glist[entrez_glist$hgnc_symbol %in% hits, "entrezgene_id"]

reactome_plot <- GSEA(glist, 0.05, top_100_genes, hits)
kegg_plot <- GSEA(glist, 0.05, top_100_genes, hits, pathway_set = "KEGG", "/mnt/sdb/Pathway_data/c2.cp.kegg_legacy.v2023.2.Hs.entrez.gmt")
wiki_plot <- GSEA(glist, 0.05, top_100_genes, hits, pathway_set = "Wiki", "/mnt/sdb/Pathway_data/c2.cp.wikipathways.v2023.2.Hs.entrez.gmt")
go_plot <- GSEA(glist, 0.05, top_100_genes, hits, pathway_set = "GO", "/mnt/sdb/Pathway_data/c5.go.v2023.2.Hs.entrez.gmt")
save(reactome_plot, kegg_plot, wiki_plot, go_plot, file="enrichments/fGSEA_plots.RData")

GSEA(glist, 1, top_100_genes, hits, 
    pathway_set = "Patel_DEG", 
    pathway_file = "/home/upamanyu/GWANN/Code_AD/post_hoc_scripts/Patel_DEG.entrez.gmt",
    maxsize = 5000)
