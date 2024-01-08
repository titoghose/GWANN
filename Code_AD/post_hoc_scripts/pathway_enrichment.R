set.seed(54321)
library(dplyr)
library(fgsea)
library(biomaRt)
setwd("/home/upamanyu/GWANN/Code_AD/results_Sens8_v4")

GSEA = function(gene_list, pval) {
    if ( any( duplicated(names(gene_list)) )  ) {
        warning("Duplicates in gene names")
        gene_list = gene_list[!duplicated(names(gene_list))]
    }
    if  ( !all( order(gene_list, decreasing = TRUE) == 1:length(gene_list)) ){
        warning("Gene list not sorted")
        gene_list = sort(gene_list, decreasing = TRUE)
    }

    pws <- reactomePathways(names(gene_list))
    # GSEA
    fgRes <- fgsea::fgseaMultilevel(pathways=pws, 
                            stats=gene_list,
                            maxSize=500) %>% 
                    as.data.frame() %>% 
                    dplyr::filter(padj < !!pval)
    fgRes <- as.data.frame(fgRes[order(fgRes$pval), ])
    fgRes$leadingEdge <- sapply(fgRes$leadingEdge, paste, collapse=";")
    View(head(fgRes))
    write.csv(fgRes, file = "enrichments/fGSEA_Reactome.csv", quote=TRUE, 
                row.names=FALSE)

    # ORA
    foRes <- fgsea::fora(pathways=pws, 
                        genes=names(gene_list[1:100]),
                        universe=names(gene_list),
                        maxSize=500) %>% 
                    as.data.frame() %>% 
                    dplyr::filter(padj < !!pval)
    foRes <- as.data.frame(foRes[order(foRes$pval), ])
    foRes$overlapGenes <- sapply(foRes$overlapGenes, paste, collapse=";")
    View(head(foRes))
    write.csv(foRes, file = "enrichments/fORA_Reactome.csv", quote=TRUE, 
                row.names=FALSE)
}

stats <- read.csv("summary_nsuperwins_1_gene_level.csv", sep=",")
glist <- as.vector(-1 * stats$stat_trial_A)
names(glist) <- stats$Gene

# Get entrez gene ids from hgnc symbols
mart <- useMart("ensembl", dataset="hsapiens_gene_ensembl")
entrez_glist <- getBM(attributes=c("hgnc_symbol", "entrezgene_id"), 
               filters="hgnc_symbol", 
               values=names(glist), 
               mart=mart)
entrez_glist <- entrez_glist[!duplicated(entrez_glist$entrezgene_id), ]

glist_m <- merge(glist, entrez_glist, by.x="row.names", by.y="hgnc_symbol")
glist_m <- glist_m[order(glist_m$x, decreasing = TRUE), ]
glist <- glist_m$x
names(glist) <- glist_m$entrezgene_id
GSEA(glist, 0.05)
