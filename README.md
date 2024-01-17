# IIRN_multi_omic

This repository contains the code associated with the paper titled "Integrative analysis of untargeted serum and fecal metabolome and gut microbiome reveals a role in Crohn Disease and with signals that precede a disease flare".

## Repository Contents

This repository contains the code for conducting the multi-omic analysis performed in the study. It includes the scripts and resources necessary to reproduce the results and further analyze the data.
For more detailed analysis and source data, please visit the paper.

predict_mtb_by_mtx.R - This code predict the metabolome based on the microbiome composition.
calculate_spls.Rmd - This code calculate sPLS between the omics.

## Usage

To use the code in this repository, follow these steps:

1. Install the required R packages specified in the session_info section.
3. Get processed data from the paper and the second github repository 
4. Set local path

## R session info
R version 4.0.2 (2020-06-22)
Platform: x86_64-w64-mingw32/x64 (64-bit)
Running under: Windows 10 x64 (build 22621)

Matrix products: default

locale:
[1] LC_COLLATE=English_Israel.1252  LC_CTYPE=English_Israel.1252    LC_MONETARY=English_Israel.1252 LC_NUMERIC=C                    LC_TIME=English_Israel.1252    

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
 [1] plotly_4.10.1      lubridate_1.8.0    forcats_1.0.0      readr_2.1.2        tidyverse_2.0.0    rstatix_0.7.2      vegan_2.6-2        permute_0.9-7     
 [9] ggdist_3.2.0       magrittr_2.0.3     stringr_1.4.0      yardstick_0.0.9    workflowsets_0.1.0 workflows_0.2.4    tune_0.1.6         tidyr_1.2.0       
[17] tibble_3.1.6       rsample_1.1.0      recipes_0.2.0      purrr_0.3.4        parsnip_0.2.0      modeldata_1.1.0    infer_1.0.4        dplyr_1.0.8       
[25] dials_0.1.0        scales_1.2.1       broom_1.0.3        tidymodels_0.1.4   mixOmics_6.14.1    ggplot2_3.3.5      lattice_0.20-45    MASS_7.3-56       

loaded via a namespace (and not attached):
 [1] readxl_1.4.0         backports_1.4.1      plyr_1.8.7           igraph_1.3.0         lazyeval_0.2.2       splines_4.0.2        BiocParallel_1.24.1 
 [8] listenv_0.9.0        digest_0.6.25        foreach_1.5.2        htmltools_0.5.2      fansi_1.0.3          cluster_2.1.3        tzdb_0.3.0          
[15] globals_0.16.2       gower_1.0.0          matrixStats_0.62.0   rARPACK_0.11-0       vroom_1.5.7          hardhat_0.2.0        colorspace_2.0-3    
[22] ggrepel_0.9.1        xfun_0.30            crayon_1.5.2         jsonlite_1.8.0       survival_3.3-1       iterators_1.0.14     glue_1.6.2          
[29] gtable_0.3.1         ipred_0.9-12         distributional_0.3.2 car_3.1-1            future.apply_1.10.0  abind_1.4-5          DBI_1.1.3           
[36] Rcpp_1.0.8.3         viridisLite_0.4.1    GPfit_1.0-8          bit_4.0.4            lava_1.7.2.1         prodlim_2019.11.13   htmlwidgets_1.5.4   
[43] httr_1.4.5           RColorBrewer_1.1-3   ellipsis_0.3.2       pkgconfig_2.0.3      farver_2.1.0         nnet_7.3-17          utf8_1.2.2          
[50] tidyselect_1.1.2     labeling_0.4.2       rlang_1.0.2          DiceDesign_1.9       reshape2_1.4.4       munsell_0.5.0        cellranger_1.1.0    
[57] tools_4.0.2          cli_3.2.0            generics_0.1.3       evaluate_0.20        fastmap_1.1.0        yaml_2.3.5           knitr_1.37          
[64] bit64_4.0.5          future_1.32.0        nlme_3.1-157         compiler_4.0.2       rstudioapi_0.14      lhs_1.1.5            stringi_1.7.6       
[71] RSpectra_0.16-0      Matrix_1.4-1         vctrs_0.4.1          pillar_1.8.1         lifecycle_1.0.1      furrr_0.3.1          data.table_1.14.2   
[78] corpcor_1.6.10       R6_2.5.1             gridExtra_2.3        parallelly_1.35.0    codetools_0.2-19     assertthat_0.2.1     withr_2.5.0         
[85] mgcv_1.8-40          parallel_4.0.2       hms_1.1.3            grid_4.0.2           rpart_4.1.16         timeDate_4022.108    class_7.3-20        
[92] rmarkdown_2.21       carData_3.0-5        pROC_1.18.0          ellipse_0.4.3       




