library(harmonicmeanp)
library(metap)

combine <- function(row, method = "hmp"){
    p_list <- as.numeric(row[-c(1, 2)])
    agg_p <- 0

    if(method == "hmp"){
        tryCatch({
            agg_p <- p.hmp(p_list, w = 1/length(p_list), L = length(p_list))
        }, 
        error = function(e){
            print(paste("Error with ", row[1], ". Setting agg_p to 0"))
        })
    }
    else if(method == "simple"){
        p_list <- sort(p_list)
        # p_list <- p_list[-c(1, length(p_list))]
        # print(length(p_list))
        
        agg_p <- 10^mean(log10(p_list))
    }
    else {
       agg_p <- min(p.adjust(p_list, method = method))
    }
    
    return(agg_p)
}

# Code to loop from 1 to 8
# for(i in 1:8){
#     print(paste("./results_Sens8_v4/results_Sens8_v4_avg/agg_P_matrix_", as.character(i), ".csv", sep = ""))
#     agg_p_matrix <- read.csv(
#         paste("./results_Sens8_v4/results_Sens8_v4_avg/agg_P_matrix_", as.character(i), ".csv", sep = ""))

#     agg_p_matrix$adj_p_simple <- p.adjust(apply(agg_p_matrix, 1, combine, method = "simple"), method = "bonferroni")
#     # agg_p_matrix$adj_p_bonf <- p.adjust(apply(agg_p_matrix, 1, combine, method = "bonferroni"), method = "bonferroni")
#     # agg_p_matrix$adj_p_hmp <- p.adjust(apply(agg_p_matrix, 1, combine, method = "hmp"), method = "bonferroni")

#     write.csv(
#         agg_p_matrix,
#         paste("./results_Sens8_v4/results_Sens8_v4_avg/agg_P_matrix_hmp_", as.character(i), ".csv", sep = ""),
#         quote = FALSE,
#         row.names = FALSE)
# }
i <- ""
print(paste("./results_Sens8_v4/results_Sens8_v4_avg/agg_P_matrix", as.character(i), ".csv", sep = ""))
agg_p_matrix <- read.csv(
    paste("./results_Sens8_v4/results_Sens8_v4_avg/agg_P_matrix", as.character(i), ".csv", sep = ""))

agg_p_matrix$adj_p_simple <- p.adjust(apply(agg_p_matrix, 1, combine, method = "simple"), method = "bonferroni")
# agg_p_matrix$adj_p_bonf <- p.adjust(apply(agg_p_matrix, 1, combine, method = "bonferroni"), method = "bonferroni")
# agg_p_matrix$adj_p_hmp <- p.adjust(apply(agg_p_matrix, 1, combine, method = "hmp"), method = "bonferroni")

write.csv(
    agg_p_matrix,
    paste("./results_Sens8_v4/results_Sens8_v4_avg/agg_P_matrix_hmp", as.character(i), ".csv", sep = ""),
    quote = FALSE,
    row.names = FALSE)