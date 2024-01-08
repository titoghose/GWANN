#
# Using double test to get stable hits from GWANN
#
# v3 = Adjusting permutations to number of windows per gene
#

# Set workspace
rm(list = ls())
library(tidyr)
library(dplyr)
library(stringr)
library(ggplot2)
library(sn)
setwd("/home/upamanyu/GWANN/Code_AD/results_Sens8_v4")
set.seed(1234)

# Load data
sFileReal = "results_Sens8_combined.csv"
sFileShuf = "results_Sens8_dummy_combined.csv"
real_tRC <- read.csv( sFileReal )
shuf_tRC <- read.csv( sFileShuf )

# shuf_0_tRC <- read.csv("results_Sens8_dummy_combined.csv")

# Divide genes into subgenes, each having a maximum of 10 superwindows
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
}

if(n_superwins != 1) {
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
johnny = function( min_real, min_shuf ){
  
  # Fit a skew-normal distribution
  min_real_2 <- ( min_real - mean( min_shuf ) ) / sd( min_shuf )
  min_shuf_2 <- ( min_shuf - mean( min_shuf ) ) / sd( min_shuf )
  fit <- selm( min_shuf_2 ~ 1 )
  p_val_real <- psn( min_real_2, xi=fit@param$dp["xi"], omega=fit@param$dp["omega"], alpha=fit@param$dp["alpha"] )
  p_val_shuf <- psn( min_shuf_2, xi=fit@param$dp["xi"], omega=fit@param$dp["omega"], alpha=fit@param$dp["alpha"] )
  
  # Some code to debug how good the fit of the skew-normal is
  hist( min_shuf_2, freq=FALSE, 100 )
  fit <- selm( min_shuf_2 ~ 1 )
  curve(dsn(x, xi=fit@param$dp["xi"], omega=fit@param$dp["omega"], alpha=fit@param$dp["alpha"]),
        from = min( min_shuf_2 ), to = max( min_shuf_2 ), col="red", add=TRUE)
  
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
rates_iRT <- data.frame( )
for( iX in 2 : round( length( seeds_u_iX ) / 2 ) ){
  
  # Run stats separately for each subgene size
  real_as_tGS <- data.frame( )
  # numWins_iW <- 1 : max( shuf_tRC$Win + 1 )
  numWins_iW <- 1 : n_superwins
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
    
    # Calculate the statistics for the real data and for the dummy data
    stopifnot( unique( table( real_f_tRC$subgene ) ) == numWins_iW[ iW ] * length( seeds_u_iX ) ) 
    stopifnot( unique( table( shuf_f_tRC$subgene ) ) == numWins_iW[ iW ] * length( seeds_u_iX ) )
    stopifnot( dim( shuf_f_tRC )[1] > 1000 )
    shuf_s_tGS <- runStat( shuf_f_tRC, seeds_u_iX, iX )
    real_s_tGS <- runStat( real_f_tRC, seeds_u_iX, iX )
    
    # Calculate permutation p-vals
    real_s_tGS$p_stat_trial_A <- johnny( min_real = real_s_tGS$stat_trial_A, 
                                         min_shuf = shuf_s_tGS$stat_trial_A )
    real_s_tGS$p_stat_trial_B <- johnny( min_real = real_s_tGS$stat_trial_B, 
                                         min_shuf = shuf_s_tGS$stat_trial_B )
    
    # Accumulate results
    real_as_tGS <- rbind( real_as_tGS,
                          real_s_tGS )
    
  }
  
  # Calculate FPR and TPR for all possible values of strong p-value threshold (i.e. p_thresold_)
  ##  Theta 1 is the first p-value threshold considered
  ##  Theta 2 is the second more permissive p-value considered
  ##  We are looking for genes that pass a very stringent theta 1 in a given test (e.g. test A), 
  ## such that we can say that in a second test (e.g. text B) those genes are very likely
  ## going to pass a more permissive theta 2 value
  theta_2 = 1e-7
  pos_A2_iG <- real_as_tGS$p_stat_trial_A < theta_2
  pos_B2_iG <- real_as_tGS$p_stat_trial_B < theta_2
  rates_0_iRT <- sapply( -40:-7,
                         function( theta_1 ){
                           pos_A1_iG <- real_as_tGS$p_stat_trial_A < 10 ^ theta_1
                           pos_B1_iG <- real_as_tGS$p_stat_trial_B < 10 ^ theta_1
                           stats_iS <- c( false_pos_rate_AB = mean( ! pos_B2_iG[ pos_A1_iG == 1 ] ),
                                          false_pos_rate_BA = mean( ! pos_A2_iG[ pos_B1_iG == 1 ] ),
                                          true_pos_rate_AB = mean( pos_B2_iG[ pos_A1_iG == 1 ] ),
                                          true_pos_rate_BA = mean( pos_A2_iG[ pos_B1_iG == 1 ] ),
                                          hits_A1 = sum( pos_A1_iG ),
                                          hits_B1 = sum( pos_B1_iG ),
                                          theta_1 = theta_1,
                                          num_seeds = iX )
                         } )
  rates_iRT <- rbind( rates_iRT,
                      t( rates_0_iRT ) )
  # View( rates_iRT )
  
  # Hale the user
  theta_1 = 1e-14
  int_iG <- real_s_tGS$p_stat_trial_A < theta_2 & real_s_tGS$p_stat_trial_B < theta_2
  uni_iG <- real_s_tGS$p_stat_trial_A < theta_2 | real_s_tGS$p_stat_trial_B < theta_2
  pos_A2_iG <- real_s_tGS$p_stat_trial_A < theta_2
  pos_B2_iG <- real_s_tGS$p_stat_trial_B < theta_2
  pos_A1_iG <- real_s_tGS$p_stat_trial_A < theta_1
  pos_B1_iG <- real_s_tGS$p_stat_trial_B < theta_1
  stopifnot( all( uni_iG[ int_iG ] == TRUE ) )
  print( paste( "==== ROUND", iX, "=====" ) )
  print( paste( "Hits =", sum(real_s_tGS$p_stat_trial_A < theta_2 ), "and", sum(real_s_tGS$p_stat_trial_B < theta_2 ), "genes" ) )
  print( paste( "Intersection =", sum( int_iG ), "genes" ) )
  print( paste( "Union =", sum( uni_iG ), "genes" ) )
  print( paste( "Int / Uni = ", sum( int_iG ) / sum( uni_iG ), "genes" ) )
  print( paste( "Hits strong =", sum(real_s_tGS$p_stat_trial_A < theta_1 ), "and", sum(real_s_tGS$p_stat_trial_B < theta_1 ), "genes" ) )
  print( paste( "False positive rate 1 -> 2 =", mean( ! pos_B2_iG[ pos_A1_iG == 1 ] ) ) )
  print( paste( "False positive rate 2 -> 1 =", mean( ! pos_A2_iG[ pos_B1_iG == 1 ] ) ) )
  print( paste( "True positive rate 1 -> 2 =", mean( pos_B2_iG[ pos_A1_iG == 1 ] ) ) )
  print( paste( "True positive rate 2 -> 1 =", mean( pos_A2_iG[ pos_B1_iG == 1 ] ) ) )
  print( " " )
  
}

# Check results
View( real_as_tGS )
View( rates_iRT )
(real_as_tGS %>% arrange( p_stat_trial_A ))[40:60,c("subgene", "Chrom", "p_stat_trial_A", "p_stat_trial_B")]

# Save results
save( file = "stablity_v4.Rdata",
      list = c( "rates_iRT",
                "real_as_tGS",
                "sFileReal",
                "sFileShuf" ) )

# Plot false_pos_rate_AB and true_pos_rate_AB against theta_1
ggplot(rates_iRT, aes(x = theta_1)) +
  geom_line(aes(y = false_pos_rate_AB, color = "False Positive Rate",)) +
  geom_line(aes(y = true_pos_rate_AB, color = "True Positive Rate")) +
  facet_wrap(~num_seeds, ncol = 4) +
  labs(color = "Rate Type") +
  ylab("Rate") +
  xlab("Theta 1") +
  ggtitle("False Positive and True Positive Rates against Theta 1") +
  theme(text = element_text(size = 20)) +  
  scale_x_continuous(breaks = seq(min(rates_iRT$theta_1), max(rates_iRT$theta_1), by = 5)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1))

ggsave(paste("FPR_TPR_vs_theta1_sw_", as.character(n_superwins), ".png", sep=""), width = 20, height = 10)

####################
# Run on all seeds #
####################
real_as_tGS <- data.frame( )
numWins_iW <- 1 : n_superwins
# Run stats separately for each subgene size
for( random_seed in c(1234) ){
  set.seed(random_seed)
  seeds_u_iX <- sample( unique( shuf_tRC[ , "Seed" ] ) )
  for( iX in 16 : length( seeds_u_iX ) ){  
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
      
      # Calculate the statistics for the real data and for the dummy data
      stopifnot( unique( table( real_f_tRC$subgene ) ) == numWins_iW[ iW ] * length( seeds_u_iX ) ) 
      stopifnot( unique( table( shuf_f_tRC$subgene ) ) == numWins_iW[ iW ] * length( seeds_u_iX ) )
      stopifnot( dim( shuf_f_tRC )[1] > 1000 )
      shuf_s_tGS <- runStat( shuf_f_tRC, seeds_u_iX, iX, single_set = TRUE )
      real_s_tGS <- runStat( real_f_tRC, seeds_u_iX, iX, single_set = TRUE )
      
      # Calculate permutation p-vals
      real_s_tGS$p_stat_trial_A <- johnny( min_real = real_s_tGS$stat_trial_A, 
                                            min_shuf = shuf_s_tGS$stat_trial_A )
      real_s_tGS$num_seeds <- iX
      real_s_tGS$random_seed <- random_seed
        
      # Accumulate results
      real_as_tGS <- rbind( real_as_tGS,
                            real_s_tGS )
      
    }
  }
}

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
  mutate( theta_1 = 10^(-1*theta_1) )
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
  xlab("Number of seeds") +
  labs(color = "theta_1") +
  ggtitle("Hit count against number of seeds") +
  theme(text = element_text(size = 20)) +
  scale_x_continuous(breaks = seq(min(hit_summary_stats$num_seeds), max(hit_summary_stats$num_seeds), by = 2))

# save plot
ggsave(paste("hit_count_vs_num_seeds_sw_", as.character(n_superwins), ".png", sep=""), width = 10, height = 10)

theta_1 <- 10^(-1*23)
hits_all_seeds <- real_as_tGS %>%
                    filter( num_seeds == 16 ) %>%
                    filter( p_stat_trial_A < theta_1)
write.csv(real_as_tGS  %>% filter( num_seeds == 16 ), 
          paste("summary_nsuperwins_", as.character(n_superwins), ".csv", sep=""),
          row.names = FALSE,
          quote = FALSE)
write.csv(hits_all_seeds, 
          paste("hits_nsuperwins_", as.character(n_superwins), ".csv", sep=""),
          row.names = FALSE,
          quote = FALSE)
                    

a <- read.csv("hits_nsuperwins_1.csv")
b <- read.csv("hits_nsuperwins_10.csv")
a <- a %>% 
      mutate( Gene = str_split_fixed(subgene, "_", 2)[,1] ) %>% 
      filter( Chrom != 19 )
b <- b %>% 
      mutate( Gene = str_split_fixed(subgene, "_", 2)[,1] ) %>% 
      filter( Chrom != 19 )

a_genes <- unique(a$Gene)
b_genes <- unique(b$Gene)
common_genes <- intersect(a_genes, b_genes)

print(paste("Number of hits using superwin size 1:", length(a_genes)))
print(paste("Number of hits using superwin size 10:", length(b_genes)))
print(paste("Number of common hits:", length(common_genes)))
setdiff(a_genes, b_genes)

# Plot venn diagram of hits using superwin size 1 and 10
library(VennDiagram)
venn.diagram(
  x = list(
    "Superwin size 1" = a_genes,
    "Superwin size 10" = b_genes
  ),
  filename = "venn_diagram_hits.png",
  output = TRUE,
  imagetype = "png",
  height = 800,
  width = 800,
  resolution = 80,
  lwd = 1,
  col = "black",
  fill = c("cornflowerblue", "green"),
  alpha = 0.50,
  cex = 1.5,
  fontfamily = "serif",
  cat.col = c("darkblue", "darkgreen"),
  cat.cex = 1.5,
  cat.fontfamily = "serif",
  cat.default.pos = "outer",
  margin = 0.05,
  main = "Venn diagram of hits using superwin size 1 and 10"
)

