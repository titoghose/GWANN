#
# Using double test to get stable hits from GWANN
#

# Set workspace
rm(list = ls())
library(tidyr)
library(dplyr)
library(stringr)
library(ggplot2)
library(sn)
library(forecast)
library(gridExtra)
library(Cairo)
set.seed(5678)

# Load white-british data
# setwd("/home/upamanyu/GWANN/Code_AD/results_Sens8_v4")
# sFileReal = "results_Sens8_combined.csv"
# sFileShuf = "results_Sens8_dummy_combined.csv"

# Load asian data
setwd("/home/upamanyu/GWANN/Code_AD/results_non_white")
sFileReal = "Asian_results_top100.csv"
sFileShuf = "Asian_results_Dummy.csv"

# Load black data
# setwd("/home/upamanyu/GWANN/Code_AD/results_Sens8_v4")
# sFileReal = "Black_results_top100.csv"
# sFileShuf = "Black_results_Dummy.csv"

real_tRC <- read.csv( sFileReal )
shuf_tRC <- read.csv( sFileShuf )

# Divide genes into subgenes, each having a maximum of 10 superwindows
# set theta2 to 0.05/70848 i.e. bonferroni for total number of windows
theta_2 <- 7.05736224028907e-07
n_superwins <- 1

addInfo = function( data_tRC ){
  sRegex <- "^([A-Za-z0-9\\.\\-]*)_([0-9]*)$"
  stopifnot( all( grepl( sRegex, data_tRC$Gene ) ) )
  data_tRC <- data_tRC %>%
    mutate( Gene = gsub( sRegex, "\\1", data_tRC$Gene ),
            Win = as.numeric( gsub( sRegex, "\\2", data_tRC$Gene ) ) ) %>%
    mutate( superwin = floor( Win / n_superwins ),
            subgene = paste( Gene, 
                             superwin, 
                             sep = "_" ) )
  return( data_tRC )
}
real_tRC = addInfo( data_tRC = real_tRC )
shuf_tRC = addInfo( data_tRC = shuf_tRC )

if(n_superwins == 1){
  stopifnot(all(real_tRC$Win == real_tRC$superwin))
  stopifnot(all(shuf_tRC$Win == shuf_tRC$superwin))
} else {
  stopifnot(any(real_tRC$Win != real_tRC$superwin))
  stopifnot(any(shuf_tRC$Win != shuf_tRC$superwin))
}

# A function to calculate the statistic
runStat = function( data_tRC, seeds_u_iX, iX, single_set = FALSE){
  
  # Group the data "data_tRC" into 2 independent trials depending on the random seed "seeds_u_iX". 
  # These sets will model "trial 1" and "trial 2"
  if(single_set){
    print(paste( "All", iX, "seeds are being considered to generate a single statistic" ))
    shuf_s_tGS <- data_tRC %>% 
      mutate( gA = Seed %in% seeds_u_iX[ 1 : iX ] ) 

    shuf_s_tGS <- shuf_s_tGS %>%
      filter( gA == 1 ) %>%
      group_by( subgene, Chrom ) %>%
      summarise( stat_trial_A = quantile( Loss[ gA == 1 ], 0.2 ),
                count_A = sum( gA == 1 ))
  }
  else{
    shuf_s_tGS <- data_tRC %>% 
      mutate( gA = Seed %in% seeds_u_iX[ 1 : iX ],
              gB = Seed %in% seeds_u_iX[ ( length( seeds_u_iX ) - iX + 1 ) : length( seeds_u_iX ) ] ) 
      stopifnot( all( !( shuf_s_tGS$gA & shuf_s_tGS$gB ) ) )
      stopifnot( sum( shuf_s_tGS$gA ) == sum( shuf_s_tGS$gB ) )
      shuf_s_tGS <- shuf_s_tGS %>%
        filter( gA == 1 | gB == 1 ) %>%
        group_by( subgene, Chrom ) %>%
        summarise( stat_trial_A = quantile( Loss[ gA == 1 ], 0.2 ),
                  #stat_trial_A = mean( sort( Loss[ gA == 1 ] )[2:max(2,n()/10)] ),
                  # stat_trial_A = mean( sort( Loss[ gA == 1 ] )[2:iX] ),
                  stat_trial_B = quantile( Loss[ gB == 1 ], 0.2 ),
                  # stat_trial_B = mean( sort( Loss[ gB == 1 ] )[2:max(2,n()/10)] ),
                  # stat_trial_B = mean( sort( Loss[ gB == 1 ] )[2:iX] ),
                  count_A = sum( gA == 1 ),
                  count_B = sum( gB == 1 ) )
    stopifnot( all( shuf_s_tGS$count_A == shuf_s_tGS$count_B ) )
  }
  
  # # Calculate the statistics for "trial 1" and "trial 2" using min(seeds) and percentile(window)
  # shuf_s_tGS <- shuf_s_tGS %>%
  #   filter( gA == 1 | gB == 1 ) %>%
  #   group_by( Gene, Win ) %>%
  #   summarise( stat_trial_A = min( Loss[ gA == 1 ] ),
  #              stat_trial_B = min( Loss[ gB == 1 ] ) ) %>%
  #   ungroup( ) %>%
  #   group_by( Gene ) %>%
  #   summarise( stat_trial_A = quantile( stat_trial_A, 0.1 ),
  #              stat_trial_B = quantile( stat_trial_B, 0.1 ) )
  
  # Calculate the statistic using the mean of the tail, with the tail being the 10% smallest values
  
  
  # Return the result
  return( shuf_s_tGS )
}

# Function to fit a distribution prob function to the dummy statistics and calculate the p-value from there
calcP = function( min_real, min_shuf, plot_dist = TRUE){
  min_real_2 <- ( min_real - mean( min_shuf ) ) / sd( min_shuf )
  min_shuf_2 <- ( min_shuf - mean( min_shuf ) ) / sd( min_shuf )
  hist( min_shuf_2, freq=FALSE, 100, xlim=c(-4, 4), col=rgb(0.5, 0.5, 0.5, alpha=0.3) )
  
  if (plot_dist){
    n_samples <- c(200, 400, 600, 800, 1000)
    colors <- c("orange", "green", "purple", "blue", "red")
    for (ni in 1:length(n_samples)){
      # Sample n_samples from the min_shuf_2 distribution
      set.seed(1234)
      min_shuf_2_sample <- sample(min_shuf_2, n_samples[ni])
      fit <- selm( min_shuf_2_sample ~ 1 )
      # curve(dsn(x, xi=fit@param$dp["xi"], omega=fit@param$dp["omega"], alpha=fit@param$dp["alpha"]),
      #       from = min( min_shuf_2 ), to = max( min_shuf_2 ), col="red", add=TRUE,)
      curve(dsn(x, xi=fit@param$dp["xi"], omega=fit@param$dp["omega"], alpha=fit@param$dp["alpha"]),
            from = -4, to = 4, col=colors[ni], add=TRUE)
    }
    legend("topright", legend = paste("Num dummy metrics =", n_samples), col = colors, lty=1)
  }
  
  # Fit a skew-normal distribution
  fit <- selm( min_shuf_2 ~ 1 )
  p_val_real <- psn( min_real_2, xi=fit@param$dp["xi"], omega=fit@param$dp["omega"], alpha=fit@param$dp["alpha"] )
  p_val_shuf <- psn( min_shuf_2, xi=fit@param$dp["xi"], omega=fit@param$dp["omega"], alpha=fit@param$dp["alpha"] )
  
  # Check the fit look like the probabilities of dummy data
  ## The first aren't too good, but at the end of the day the permutation test is simply changing 
  ## the scale of the statistic from something uninterpretable to something that can be interpreted
  ## as p-values. Namely, the order of the values do not change.
  stopifnot( abs( mean( p_val_shuf < 0.1 ) - 0.1 ) < 0.2 )
  stopifnot( abs( mean( p_val_shuf < 0.2 ) - 0.2 ) < 0.2 )
  stopifnot( abs( mean( p_val_shuf < 0.3 ) - 0.3 ) < 0.2 )
  
  # Return pvals
  return( p_val_real )
}

# Get gene level info
subgeneInfo_iG2 <- real_tRC %>%
  group_by( subgene ) %>%
  summarise( numWins = length( unique( Win ) ) )

# Run stats for each number of seeds
seeds_u_iX <- sample( unique( shuf_tRC[ , "Seed" ] ) )
stopifnot( setequal( seeds_u_iX, unique( real_tRC[ , "Seed" ] ) ) )

#####################
# Run FPR-TPR check #
#####################
# rates_iRT <- data.frame( )
# for (dummy_win in 0:9){
#   for(random_seed in c(12, 34, 56, 78)){
#     set.seed(random_seed)
#     seeds_u_iX <- sample( unique( shuf_tRC[ , "Seed" ] ) )
#     stopifnot( setequal( seeds_u_iX, unique( real_tRC[ , "Seed" ] ) ) )
#     for( iX in 2 : round( length( seeds_u_iX ) / 2 ) ){
      
#       # Run stats separately for each subgene size
#       real_as_tGS <- data.frame( )
#       # numWins_iW <- 1 : max( shuf_tRC$Win + 1 )
#       numWins_iW <- 1 : n_superwins
#       for( iW in 1 : length( numWins_iW ) ){
        
#         # Hale the user
#         print( paste( "Running stats for", iX, "seeds on subgenes of size", numWins_iW[iW] ) )
      
#         # Take only the superwindows with numWins[ iW ] windows
#         subgene_iS <- subgeneInfo_iG2 %>%
#           filter( numWins == numWins_iW[ iW ] ) %>%
#           pull( subgene )
#         real_f_tRC <- real_tRC %>%
#           filter( subgene %in% subgene_iS )
        
#         # Crop the shuffled superwindows to numWins[ iW ] windows each
#         shuf_f_tRC <- shuf_tRC %>%
#           filter( Win == dummy_win )
        
#         # Calculate the statistics for the real data and for the dummy data
#         stopifnot( unique( table( real_f_tRC$subgene ) ) == numWins_iW[ iW ] * length( seeds_u_iX ) ) 
#         stopifnot( unique( table( shuf_f_tRC$subgene ) ) == numWins_iW[ iW ] * length( seeds_u_iX ) )
#         stopifnot( dim( shuf_f_tRC )[1] > 1000 )
#         shuf_s_tGS <- runStat( shuf_f_tRC, seeds_u_iX, iX )
#         real_s_tGS <- runStat( real_f_tRC, seeds_u_iX, iX )
        
#         # Calculate permutation p-vals
#         real_s_tGS$p_stat_trial_A <- calcP( min_real = real_s_tGS$stat_trial_A, 
#                                             min_shuf = shuf_s_tGS$stat_trial_A )
#         real_s_tGS$p_stat_trial_B <- calcP( min_real = real_s_tGS$stat_trial_B, 
#                                             min_shuf = shuf_s_tGS$stat_trial_B )
        
#         # Accumulate results
#         real_as_tGS <- rbind( real_as_tGS,
#                               real_s_tGS )
        
#       }
      
#       # Calculate FPR and TPR for all possible values of strong p-value threshold (i.e. p_thresold_)
#       ##  Theta 1 is the first p-value threshold considered
#       ##  Theta 2 is the second more permissive p-value considered
#       ##  We are looking for genes that pass a very stringent theta 1 in a given test (e.g. test A), 
#       ## such that we can say that in a second test (e.g. text B) those genes are very likely
#       ## going to pass a more permissive theta 2 value
#       pos_A2_iG <- real_as_tGS$p_stat_trial_A < theta_2
#       pos_B2_iG <- real_as_tGS$p_stat_trial_B < theta_2
#       rates_0_iRT <- sapply( -40:-7,
#                             function( theta_1 ){
#                               pos_A1_iG <- real_as_tGS$p_stat_trial_A < 10 ^ theta_1
#                               pos_B1_iG <- real_as_tGS$p_stat_trial_B < 10 ^ theta_1
#                               stats_iS <- c( false_pos_rate_AB = mean( ! pos_B2_iG[ pos_A1_iG == 1 ] ),
#                                               false_pos_rate_BA = mean( ! pos_A2_iG[ pos_B1_iG == 1 ] ),
#                                               true_pos_rate_AB = mean( pos_B2_iG[ pos_A1_iG == 1 ] ),
#                                               true_pos_rate_BA = mean( pos_A2_iG[ pos_B1_iG == 1 ] ),
#                                               hits_A1 = sum( pos_A1_iG ),
#                                               hits_B1 = sum( pos_B1_iG ),
#                                               theta_1 = theta_1,
#                                               num_seeds = iX,
#                                               random_seed = random_seed )
#                             } )
#       rates_iRT <- rbind( rates_iRT,
#                           t( rates_0_iRT ) )
#       # View( rates_iRT )
      
#       # Hale the user
#       theta_1 = 1e-14
#       int_iG <- real_s_tGS$p_stat_trial_A < theta_2 & real_s_tGS$p_stat_trial_B < theta_2
#       uni_iG <- real_s_tGS$p_stat_trial_A < theta_2 | real_s_tGS$p_stat_trial_B < theta_2
#       pos_A2_iG <- real_s_tGS$p_stat_trial_A < theta_2
#       pos_B2_iG <- real_s_tGS$p_stat_trial_B < theta_2
#       pos_A1_iG <- real_s_tGS$p_stat_trial_A < theta_1
#       pos_B1_iG <- real_s_tGS$p_stat_trial_B < theta_1
#       stopifnot( all( uni_iG[ int_iG ] == TRUE ) )
#       print( paste( "==== ROUND", iX, "=====" ) )
#       print( paste( "Hits =", sum(real_s_tGS$p_stat_trial_A < theta_2 ), "and", sum(real_s_tGS$p_stat_trial_B < theta_2 ), "genes" ) )
#       print( paste( "Intersection =", sum( int_iG ), "genes" ) )
#       print( paste( "Union =", sum( uni_iG ), "genes" ) )
#       print( paste( "Int / Uni = ", sum( int_iG ) / sum( uni_iG ), "genes" ) )
#       print( paste( "Hits strong =", sum(real_s_tGS$p_stat_trial_A < theta_1 ), "and", sum(real_s_tGS$p_stat_trial_B < theta_1 ), "genes" ) )
#       print( paste( "False positive rate 1 -> 2 =", mean( ! pos_B2_iG[ pos_A1_iG == 1 ] ) ) )
#       print( paste( "False positive rate 2 -> 1 =", mean( ! pos_A2_iG[ pos_B1_iG == 1 ] ) ) )
#       print( paste( "True positive rate 1 -> 2 =", mean( pos_B2_iG[ pos_A1_iG == 1 ] ) ) )
#       print( paste( "True positive rate 2 -> 1 =", mean( pos_A2_iG[ pos_B1_iG == 1 ] ) ) )
#       print( " " )
      
#     }
#   }

#   # Save results
#   save( file = paste(as.character(dummy_win), "_stablity_v4.Rdata"),
#         list = c( "rates_iRT",
#                   "real_as_tGS",
#                   "sFileReal",
#                   "sFileShuf" ) )

#   plot_df <- rates_iRT %>%
#               mutate(fpr_AB_BA = (false_pos_rate_AB + false_pos_rate_BA)/2) %>%
#               mutate(tpr_AB_BA = (true_pos_rate_AB + true_pos_rate_BA)/2)
#   plot_df_stats <- plot_df %>%
#                     group_by(num_seeds, theta_1) %>%
#                     summarise(mean_fpr = mean(fpr_AB_BA), sd_fpr = sd(fpr_AB_BA), min_fpr = min(fpr_AB_BA), max_fpr = max(fpr_AB_BA),
#                               mean_tpr = mean(tpr_AB_BA), sd_tpr = sd(tpr_AB_BA), min_tpr = min(tpr_AB_BA), max_tpr = max(tpr_AB_BA))

#   # Plot false_pos_rate_AB and true_pos_rate_AB against theta_1 with errorlines
#   stability_plot_1 <- ggplot(plot_df_stats, aes(x = theta_1)) +
#     geom_line(aes(y = mean_fpr)) +
#     geom_ribbon(aes(ymin = min_fpr, ymax = max_fpr, fill = "Hit instability rate"), alpha=0.3) +
#     geom_line(aes(y = mean_tpr)) +
#     geom_ribbon(aes(ymin = min_tpr, ymax = max_tpr, fill = "Hit stability rate"), alpha=0.3) +
#     facet_wrap(~num_seeds, ncol = 3, labeller = labeller(num_seeds = function(x) paste("Number of runs:", x))) +
#     labs(fill = "Rate Type") +
#     ylab("Rate") +
#     xlab(expression(paste("log"[10], "(Empirical P-value threshold)"))) +
#     # ggtitle("Hit  Theta 1") +
#     theme_bw() +
#     theme(text = element_text(size = 20), plot.margin = margin(1, 1, 1, 1, 'cm')) +  
#     scale_x_continuous(breaks = seq(min(rates_iRT$theta_1), max(rates_iRT$theta_1), by = 5)) +
#     scale_y_continuous(breaks = seq(0, 1, by = 0.1))

#   # For rates_iRT, plot the value of the theta_1 column closest to the column true_pos_rate_AB of 0.95
#   # against num_seeds
#   tpr90_theta1 <- rates_iRT %>%
#               mutate(mean_tpr = (true_pos_rate_AB + true_pos_rate_BA)/2) %>%
#               group_by(num_seeds, random_seed) %>%
#               filter(mean_tpr >= 0.9) %>%
#               summarise(theta_1 = max(theta_1))
#   tpr90_theta1_stats <- tpr90_theta1 %>%
#                         group_by(num_seeds) %>%
#                         summarise(mean_theta_1 = mean(theta_1), sd_theta_1 = sd(theta_1))

#   stability_plot_2 <- ggplot(tpr90_theta1_stats, aes(x = num_seeds, y = mean_theta_1)) +
#     geom_line() +
#     geom_point() +
#     geom_errorbar(aes(ymin = mean_theta_1 - sd_theta_1, ymax = mean_theta_1 + sd_theta_1), width = 0.2) +
#     ylab(expression(paste("log"[10], "(Empirical P-value threshold)"))) +
#     xlab("Number of runs") +
#     # ggtitle("Theta 1 against number of runs to get Stability>=0.90") +
#     theme_bw() +
#     theme(text = element_text(size = 20), plot.margin = margin(1, 1, 1, 1, 'cm')) +
#     scale_x_continuous(breaks = seq(min(tpr90_theta1_stats$num_seeds), max(tpr90_theta1_stats$num_seeds), by = 1))

#   combined_plot <- grid.arrange(stability_plot_1, stability_plot_2, ncol = 2, widths = c(3, 2))
#   ggsave(paste("stability_plots/", as.character(dummy_win), "_Stability_plot_sw_", as.character(n_superwins), ".png", sep=""), 
#           plot=combined_plot, width = 20, height = 10)
#   ggsave(paste("stability_plots/", as.character(dummy_win), "_Stability_plot_sw_", as.character(n_superwins), ".svg", sep=""), 
#           plot=combined_plot, width = 20, height = 10)
# }

# ####################
# # Run on all seeds #
# ####################
dummy_number <- ""
real_as_tGS <- data.frame( )
numWins_iW <- 1 : n_superwins
sd_df <- data.frame()
# Run stats separately for each subgene size
# c(12, 34, 56, 78)
for( random_seed in c(34) ){
  set.seed(random_seed)
  seeds_u_iX <- sample( unique( shuf_tRC[ , "Seed" ] ) )
  sds <- c()
  for( iX in 16:16 ){  
    for( iW in 1 : length( numWins_iW ) ){
      
      # Hale the user
      print( paste( "Running stats for", iX, "seeds on subgenes of size", numWins_iW[iW] ) )

      # Take only the superwindows with numWins[ iW ] windows
      subgene_iS <- subgeneInfo_iG2 %>%
        filter( numWins == numWins_iW[ iW ] ) %>%
        pull( subgene )
      real_f_tRC <- real_tRC %>%
        filter( subgene %in% subgene_iS )
      
      # Crop the shuffled superwindows to numWins[ iW ] windows each
      shuf_f_tRC <- shuf_tRC %>%
        filter( Win < numWins_iW[ iW ] )
      # shuf_f_tRC <- shuf_tRC %>%
      #   filter( Win == dummy_number )

      # Calculate the statistics for the real data and for the dummy data
      stopifnot( unique( table( real_f_tRC$subgene ) ) == numWins_iW[ iW ] * length( seeds_u_iX ) ) 
      stopifnot( unique( table( shuf_f_tRC$subgene ) ) == numWins_iW[ iW ] * length( seeds_u_iX ) )
      stopifnot( dim( shuf_f_tRC )[1] > 1000 )
      shuf_s_tGS <- runStat( shuf_f_tRC, seeds_u_iX, iX, single_set = TRUE )
      real_s_tGS <- runStat( real_f_tRC, seeds_u_iX, iX, single_set = TRUE )
      
      sds <- cbind(sds, sd(shuf_s_tGS$stat_trial_A))

      # Calculate permutation p-vals
      if (iX == length(seeds_u_iX)){
        CairoPNG(paste("stability_plots/dummy_dist_16runs.png", sep=""), 
            width = 10, height = 10, units = "in", res=300)
        real_s_tGS$p_stat_trial_A <- calcP( min_real = real_s_tGS$stat_trial_A, 
                                            min_shuf = shuf_s_tGS$stat_trial_A )
        dev.off()
      } else {
        real_s_tGS$p_stat_trial_A <- calcP( min_real = real_s_tGS$stat_trial_A, 
                                            min_shuf = shuf_s_tGS$stat_trial_A )
      }
      
      real_s_tGS$num_seeds <- iX
      real_s_tGS$random_seed <- random_seed
        
      # Accumulate results
      real_as_tGS <- rbind( real_as_tGS,
                            real_s_tGS )
      
    }
  }
  seed_sd_df <- data.frame(cbind(as.vector(sds), seq(2, length(sds)+1)))
  colnames(seed_sd_df) <- c("sds", "num_seeds")
  seed_sd_df$random_seed <- random_seed
  sd_df <- rbind(sd_df, seed_sd_df)
}

# plot sds against number of seeds for sd_df
sd_df_stats <- sd_df %>%
                  group_by(num_seeds) %>%
                  summarise(mean_sd = mean(sds), sd_sd = sd(sds))
ggplot(sd_df_stats, aes(x = num_seeds, y = mean_sd)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(ymin = mean_sd - sd_sd, ymax = mean_sd + sd_sd), width = 0.2) +
  ylab("Standard deviation of null distribution") +
  xlab("Number of runs") +
  # ggtitle("Standard deviation against number of seeds") +
  theme_bw() +
  theme(text = element_text(size = 20)) +
  scale_x_continuous(breaks = seq(min(sd_df$num_seeds), max(sd_df$num_seeds), by = 2))
ggsave(paste("stability_plots/sd_vs_num_runs_sw_", as.character(n_superwins), ".png", sep=""), width = 10, height = 10)

hit_summary <- data.frame( )
for( theta_1 in seq(20, 40, 2) ){
  hits_at_theta <- real_as_tGS %>%
    group_by( num_seeds, random_seed ) %>%
    summarise( hits = sum( p_stat_trial_A < 10^(-1*theta_1) ) )
  hit_summary <- rbind( hit_summary,
                        data.frame( hits_at_theta,
                                    theta_1 = theta_1 ) )
}
hit_summary <- hit_summary %>%
  mutate( theta_1 = -1*theta_1 )
View( hit_summary )

# Plot number of hits vs number of seeds for different theta_1
hit_summary_stats <- hit_summary %>%
  group_by(num_seeds, theta_1) %>%
  summarise(mean_hits = mean(hits), sd_hits = sd(hits))
ggplot(hit_summary_stats, aes(x = num_seeds, y = mean_hits, color = as.factor(theta_1))) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(ymin = mean_hits - sd_hits, ymax = mean_hits + sd_hits), width = 0.2) +
  ylab("Number of hits") +
  xlab("Number of runs") +
  labs(color = expression(paste("log"[10], "(Empirical P-value threshold)"))) +
  # ggtitle("Hit count against number of runs") +
  theme_bw() +
  theme(text = element_text(size = 20), legend.position = "bottom") +
  guides(color = guide_legend(nrow = 3, byrow = TRUE)) +
  scale_x_continuous(breaks = seq(min(hit_summary_stats$num_seeds), max(hit_summary_stats$num_seeds), by = 2))
ggsave(paste("stability_plots/hit_count_vs_num_runs_sw_", as.character(n_superwins), ".png", sep=""), width = 10, height = 10)

# Set theta_1 to the threshold that gives TPR >= 0.95 for 8 seeds
# This will ensure that the TPR will definitely be >= 0.95 for 16 seeds
theta_1 <- 10^(-1*25)
real_as_tGS_unique <- real_as_tGS %>% 
                filter( num_seeds == 16 ) %>% 
                distinct(subgene, .keep_all = TRUE)
hits_all_seeds <- real_as_tGS_unique %>%
                    filter( p_stat_trial_A < theta_1)

# filter real_as_tGS for num_seeds == 16 and drop duplicates of subgene
write.csv(real_as_tGS_unique, 
          paste("summary_nsuperwins_", as.character(n_superwins), as.character(dummy_number), ".csv", sep=""),
          row.names = FALSE,
          quote = FALSE)
write.csv(hits_all_seeds, 
          paste("hits_nsuperwins_", as.character(n_superwins), as.character(dummy_number), ".csv", sep=""),
          row.names = FALSE,
          quote = FALSE)

# Get theta_1 for TPR >= 0.95 for different null distributions
# thetas <- c()
# for (dummy_win in 0:9){
#   load( file = paste(as.character(dummy_win), "_stablity_v4.Rdata") )
#   plot_df <- rates_iRT %>%
#               mutate(fpr_AB_BA = (false_pos_rate_AB + false_pos_rate_BA)/2) %>%
#               mutate(tpr_AB_BA = (true_pos_rate_AB + true_pos_rate_BA)/2)
#   plot_df_stats <- plot_df %>%
#                     group_by(num_seeds, theta_1) %>%
#                     summarise(mean_fpr = mean(fpr_AB_BA), sd_fpr = sd(fpr_AB_BA), min_fpr = min(fpr_AB_BA), max_fpr = max(fpr_AB_BA),
#                               mean_tpr = mean(tpr_AB_BA), sd_tpr = sd(tpr_AB_BA), min_tpr = min(tpr_AB_BA), max_tpr = max(tpr_AB_BA))
#   tpr95_theta1 <- plot_df_stats %>%
#                     filter(num_seeds == 8) %>%
#                     group_by(num_seeds) %>%
#                     filter(mean_tpr >= 0.95) %>%
#                     summarise(theta_1 = max(theta_1))
#   thetas <- c(thetas, tpr95_theta1$theta_1)
# }

