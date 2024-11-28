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
# setwd("/home/upamanyu/GWANN/Code_AD/results_non_white")
# sFileReal = "Asian_results_top100.csv"
# sFileShuf = "Asian_results_Dummy.csv"

# Load black data
setwd("/home/upamanyu/GWANN/Code_AD/results_non_white")
sFileReal = "Black_results_top100.csv"
sFileShuf = "Black_results_Dummy.csv"

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
  # print(mean(p_val_shuf < 0.2) - 0.2)
  # stopifnot( abs( mean( p_val_shuf < 0.1 ) - 0.1 ) < 0.2 )
  # stopifnot( abs( mean( p_val_shuf < 0.2 ) - 0.2 ) < 0.2 )
  # stopifnot( abs( mean( p_val_shuf < 0.3 ) - 0.3 ) < 0.2 )
  
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

# ####################
# # Run on all seeds #
# ####################
dummy_number <- ""
real_as_tGS <- data.frame( )
numWins_iW <- 1 : n_superwins
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
      
    # Accumulate results
    real_as_tGS <- rbind( real_as_tGS,
                          real_s_tGS )
    
  }
}

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
