## Predicting the Popularity of Online News
Collaborative project between Ilana Feldman and Jasmine Wang.

### 1. Purpose of the repo
  
The purpose of the repo is to use supervised machine learning methods (linear regressions, ensemble methods) to develop different predictive models that focus on predicting the popularity of online news articles given the features of those articles published on Mashable, one of the largest online news sites. Since the target response is a continuous variable, the number of shares, the predictive models are compared with their RMSE value when fitting on the test set. Then, the best predictive model is chosen with the lowest RMSE value when fitting on the test set. We declare a winning model in predicting the online news popularity measures. In addition, this process is automated to run the analysis, perform exploratory data analysis, fit different predictive models, compare different predictive models and find the optimal model for each of the data channels. The articles were collected from six data channels on Mashable: lifestyle, entertainment, business, social media, technology and world.  
  
### 2. A list of R packages used to generate the analyses

   * rmarkdown
   * tidyverse
   * knitr
   * caret
   * corrplot
   * ggplot2
   * gbm
   * vip

### 3. Links to each of the generated analyses using automation

   * The analysis for [Lifestyle articles is available here](lifestyle.html)
   * The analysis for [Entertainment articles is available here](entertainment.html)
   * The analysis for [Business articles is available here](bus.html)
   * The analysis for [Social Media articles is available here](socmed.html)
   * The analysis for [Technology articles is available here](tech.html)
   * The analysis for [World articles is available here](world.html)

### 4. The render function used to create 6 analyses for the channels

```markdown  
  library(rmarkdown)  
  library(tidyverse)  
  type <- c("lifestyle", "entertainment", "bus", "socmed", "tech", "world")  
  output_file <- paste0(type, ".md")  
  params <- lapply(type, FUN = function(x){list(channel = x)})  
  reports <- tibble(output_file, params)  
  
  apply(reports, MARGIN = 1,  
        FUN = function(x){  
          render(input = "C:/Users/peach/Documents/ST558/ST558_repos/ST558-Project-2/_Rmd/ST558_project2_auto.Rmd",  
                 output_format = "github_document",  
                 output_file = paste0("C:/Users/peach/documents/ST558/ST558_repos/ST558-Project-2/", x[[1]]),  
                 params = x[[2]],  
                 output_options = list(html_preview = FALSE, toc = TRUE, toc_depth = 3, df_print = "tibble"))  
        })  
```  
