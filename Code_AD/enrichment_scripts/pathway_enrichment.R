GSEA = function(gene_list, pval) {
    set.seed(54321)
    library(dplyr)
    library(fgsea)

    if ( any( duplicated(names(gene_list)) )  ) {
    warning("Duplicates in gene names")
    gene_list = gene_list[!duplicated(names(gene_list))]
    }
    if  ( !all( order(gene_list, decreasing = TRUE) == 1:length(gene_list)) ){
        warning("Gene list not sorted")
        gene_list = sort(gene_list, decreasing = TRUE)
    }

    pws <- reactomePathways(names(gene_list))
    fgRes <- fgsea::fgseaMultilevel(pathways=pws, 
                            stats=gene_list,
                            maxSize=500,
                            scoreType="pos") %>% 
                    as.data.frame() 
                    %>% 
                    dplyr::filter(padj < !!pval)
    fgRes <- as.data.frame(fgRes[order(fgRes$pval), ])
    print(head(fgRes))
    write.csv(fgRes, file = "../results_Sens8_00_GS10_v4/enrichments/fGSEA_Reactome.csv", quote=FALSE)

}

stats <- read.csv("/home/upamanyu/GWANN/Code_AD/results_Sens8_00_GS10_v4/FH_AD_Loss_Sens8_00_GS10_v4_gene_summary.csv", sep=",")
stats <- stats[!(stats$P == 0),]
stats <- stats[!duplicated(stats$Gene),]
glist <- as.vector(-log10(stats$P))
names(glist) <- stats$entrez_id
glist <- glist[order(glist, decreasing=TRUE)]
GSEA(glist, 0.05)