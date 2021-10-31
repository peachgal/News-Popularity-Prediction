Predictive Models for Popularity of Online News
================
Jasmine Wang & Ilana Feldman
10/31/2021

-   [1. Introduction](#1-introduction)
-   [2. Data](#2-data)
-   [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
    -   [3.1. Numerical Summaries](#31-numerical-summaries)
    -   [3.2. Visualizations](#32-visualizations)
        -   [3.2.1. Correlation Plot](#321-correlation-plot)
        -   [3.2.2. Boxplot](#322-boxplot)
        -   [3.2.3.Barplot](#323barplot)
        -   [3.2.4. Line Plot](#324-line-plot)
        -   [3.2.5. Scatterplots](#325-scatterplots)
        -   [3.2.6. QQ Plots](#326-qq-plots)
-   [4. Candidate Models](#4-candidate-models)
    -   [4.1. Linear Regression](#41-linear-regression)
    -   [4.2.Random Forest](#42random-forest)
    -   [4.3. Boosted Tree](#43-boosted-tree)
    -   [4.4. Model Comparisons](#44-model-comparisons)
-   [5. Final Model Fit with Entire
    Data](#5-final-model-fit-with-entire-data)
-   [6. Automation](#6-automation)

# 1. Introduction

Due to the expansion of online businesses, people can almost do anything
online. With the increasing amount of Internet usages, there has been a
growing interest in online news as well since they allow for an easier
and faster spread of information around the world. Hence, predicting the
popularity of online news has become an interest for research purposes.
Popularity of online news is frequently measured by the number of
interactions in the social networks such as number of likes, comments
and shares. Predicting such measures is important to news authors,
advertisers and publishing organizations. Therefore, the study collected
news articles published between January 7th, 2013 and January 7th, 2015
from different channels on Mashable which is one of the largest online
news sites. 

The study collected a total of 39,000 news articles from six data
channel categories such as lifestyle, entertainment, business,
social media, technology and world. In addition, the features contained
in the articles were also measured to help predict the popularity of the
news contents. Such features include digital media content (number of
images or videos), earlier popularity of news referenced in the article;
average number of shares of keywords, natural language features (title
popularity or Latent Dirichlet Allocation topics) and many others. The
study included 58 predictive attributes, 2 non-predictive attributes and
1 goal field which is the number of shares of the articles. The
collected data was donated by the study to the [UCI Machine Learning
repository](https://archive.ics.uci.edu/ml/datasets/online+news+popularity)
where we downloaded the data.

Table 1 shows the list of variables we used in the analysis and their
descriptions. The study shows after the best predictive model was
selected using a test set, these variables that we are interested in are
among the top ranked features according to their importance in the final
predictive model using the entire data set. Thus, we are going to
investigate their importance in predicting the number of shares using
the predictive models we propose. 

The purpose of the analyses is to compare different predictive models
and choose the best model in predicting the popularity of online news
regarding their features in different channel categories. The methods
implemented in prediction of shares are linear regression models using
different predictors, a random forest model and a boosted tree model.
More details are in the *Candidate Models* section.

The optimal model is chosen based on the majority score of the three
metrics, root Mean Squared Error, Mean Absolute Error, and R-Squared
values when fitting the candidate models on the test set. If each of the
metrics picks a different model or the majority score is a tie, the one
with the lowest RMSE will be chosen.

Table 1. Attributes used in the analyses for prediction of online news
popularity

| Index | Attribute                    | Attribute Information                                            | Type    |
|-------|------------------------------|------------------------------------------------------------------|---------|
| 1     | `shares` (target)            | Number of shares                                                 | number  |
| 2     | `kw_avg_avg`                 | Average keyword (average shares)                                 | number  |
| 3     | `LDA_02`                     | Closeness of current article to a LDA Topic 2                    | ratio   |
| 4.1   | `weekday_is_monday`          | Was the article published on a Monday?                           | boolean |
| 4.2   | `weekday_is_tuesday`         | Was the article published on a Tuesday?                          | boolean |
| 4.3   | `weekday_is_wednesday`       | Was the article published on a Wednesday?                        | boolean |
| 4.4   | `weekday_is_thursday`        | Was the article published on a Thursday?                         | boolean |
| 4.5   | `weekday_is_friday`          | Was the article published on a Friday?                           | boolean |
| 4.6   | `weekday_is_saturday`        | Was the article published on a Saturday?                         | boolean |
| 4.7   | `weekday_is_sunday`          | Was the article published on a Sunday?                           | boolean |
| 5     | `self_reference_avg_sharess` | Avg. shares of earlier popularity news referenced in the article | number  |
| 6     | `average_token_length`       | Average length of the words in the content                       | number  |
| 7     | `n_tokens_content`           | Number of words in the content                                   | number  |
| 8     | `n_tokens_title`             | Number of words in the title                                     | number  |
| 9     | `global_subjectivity`        | Text subjectivity                                                | ratio   |
| 10    | `num_imgs`                   | Number of images                                                 | number  |

``` r
library(rmarkdown)
library(tidyverse)
library(knitr)
library(caret)
library(corrplot)
library(ggplot2)
library(gbm)
library(vip)

allnews <- read_csv("C:/Users/peach/Documents/ST558/ST558_repos/News-Popularity-Prediction/_Data/OnlineNewsPopularity.csv", col_names = TRUE)

########KNIT with parameters!!!!!!!!!channels is in quotes!!!!Need to use it with quotes!!!!!!!!!!!!!!!!!!!!!!!!
channels <- paste0("data_channel_is_", params$channel)
subnews <- allnews[allnews[, channels] == 1, ]

news <- subnews %>% select(
  -data_channel_is_lifestyle, -data_channel_is_entertainment, -data_channel_is_bus, -data_channel_is_socmed, 
  -data_channel_is_tech, -data_channel_is_world, -url, -timedelta)
#################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

diffday <- news %>% mutate(log.shares = log(shares),
                           class_shares = if_else(shares < 1400, 0, 1),
                           dayweek = if_else(weekday_is_monday == 1, 1,
                                    if_else(weekday_is_tuesday == 1, 2,
                                    if_else(weekday_is_wednesday == 1, 3,
                                    if_else(weekday_is_thursday == 1, 4,
                                    if_else(weekday_is_friday == 1, 5,
                                    if_else(weekday_is_saturday == 1, 6, 7))))))
                           )

sel_data <- diffday %>% select(class_shares, shares, log.shares, dayweek, 
                               kw_avg_avg, 
                               LDA_00, LDA_01, LDA_02, LDA_03, LDA_04, 
                               weekday_is_monday, weekday_is_tuesday, weekday_is_wednesday,
                               weekday_is_thursday, weekday_is_friday, weekday_is_saturday, weekday_is_sunday,
                               self_reference_avg_sharess, 
                               average_token_length, 
                               n_tokens_content, n_tokens_title, global_subjectivity, 
                               num_imgs)

set.seed(388588)
sharesIndex <- createDataPartition(sel_data$shares, p = 0.7, list = FALSE)
train <- sel_data[sharesIndex, ]
test <- sel_data[-sharesIndex, ]
train
```

    ## # A tibble: 4,382 x 23
    ##    class_shares shares log.shares dayweek kw_avg_avg LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 weekday_is_monday weekday_is_tuesday
    ##           <dbl>  <dbl>      <dbl>   <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>             <dbl>              <dbl>
    ##  1            0    711       6.57       1         0   0.800 0.0500 0.0501 0.0501 0.0500                 1                  0
    ##  2            1   3100       8.04       1         0   0.867 0.0333 0.0333 0.0333 0.0333                 1                  0
    ##  3            0    852       6.75       1         0   0.300 0.0500 0.0500 0.0500 0.550                  1                  0
    ##  4            1   3200       8.07       1         0   0.744 0.169  0.0286 0.0295 0.0286                 1                  0
    ##  5            0    575       6.35       1         0   0.441 0.0400 0.239  0.240  0.0400                 1                  0
    ##  6            0    819       6.71       1         0   0.172 0.626  0.0200 0.0206 0.161                  1                  0
    ##  7            1   2000       7.60       3       802.  0.562 0.288  0.0500 0.0500 0.0500                 0                  0
    ##  8            1   1900       7.55       3       642.  0.840 0.0400 0.0400 0.0400 0.0400                 0                  0
    ##  9            1   1900       7.55       3       955.  0.699 0.0205 0.0200 0.240  0.0200                 0                  0
    ## 10            0    648       6.47       3       930.  0.449 0.317  0.0286 0.0286 0.178                  0                  0
    ## # ... with 4,372 more rows, and 11 more variables: weekday_is_wednesday <dbl>, weekday_is_thursday <dbl>,
    ## #   weekday_is_friday <dbl>, weekday_is_saturday <dbl>, weekday_is_sunday <dbl>, self_reference_avg_sharess <dbl>,
    ## #   average_token_length <dbl>, n_tokens_content <dbl>, n_tokens_title <dbl>, global_subjectivity <dbl>, num_imgs <dbl>

``` r
train1 <- train %>% select(-class_shares, -shares, 
                           -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                           -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04)

test1 <- test %>% select(-class_shares, -shares, 
                         -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday, -weekday_is_thursday, 
                         -weekday_is_friday, -weekday_is_saturday, -weekday_is_sunday, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
```

# 2. Data

When a subset of data is selected for the bus channel articles which
contain 6258 articles, the subset of data is then split into a training
set (70% of the subset data) and a test set (30% of the subset data)
based on the target variable, the number of shares. There are 4382
articles in the training set and 1876 observations in the test set
regarding the bus channel. The `createDataPartition` function from the
`caret` package is used to split the data into training and test sets.
We set a seed so that the analyses we implemented are reproducible.

The data donated by the study contains a “day of the week” categorical
variable but in a boolean format (dummy variable) asking if the article
was published on a day of a week for all seven days which is also shown
in Table 1. Thus, we created a new variable called `dayweek` with seven
levels to combine these dummy variables for the linear regression
models. When `dayweek` = 1, the article was published on a Monday, when
`dayweek` = 2, the article was published on a Tuesday, …, and when
`dayweek` = 7, the article was published on a Sunday.

However, these `dayweek` related variables for each day of the week in
boolean format are needed when we run the ensemble models. In addition,
we classified the articles based on their number of shares into two
categories, a “popular” group when their number of shares is more than
1,400 and an “unpopular” group when their number of shares is less than
1,400. Note that, when we dichotomize a continuous variable into
different groups, we lose information about that variable. We hope to
see some patterns in different categories of shares although what we
discover may not reflect on what the data really presents because we did
not use the “complete version” of the information within the data. This
is purely for data exploratory analysis purpose in the next section.

# 3. Exploratory Data Analysis

The bus channel has 4382 articles collected. Now let us take a look at
the relationships between our response and the predictors with some
numerical summaries and plots.

## 3.1. Numerical Summaries

Table 2 shows the popularity of the news articles on different days of
the week. We classified the number of shares greater than 1400 in a day
as “popular” and the number of shares less than 1400 in a day as
“unpopular”. We can see the number of articles from the bus channel
classified into “popular” group or “unpopular” group on different days
of the week from January 7th, 2013 to January 7th, 2015 when the
articles were published and retrieved by the study. Note, this table may
not reflect on the information contained in the data due to
dichotomizing the data.

Table 3 shows the average shares of the articles on different days of
the week. We can compare and determine which day of the week has the
most average number of shares for the bus channel. Here, we can see a
potential problem for our analysis later. Median shares are all very
different from the average shares on any day of the week. Recall that
median is a robust measure for center. It is robust to outliers in the
data. On the contrary, mean is also a measure of center but it is not
robust to outliers. Mean measure can be influenced by potential
outliers.

In addition, Table 3 also shows the standard deviation of shares is huge
for any day of the week. They are potentially larger than the average
shares. This tells us the variance of shares for any day is huge. We
know a common variance stabilizing transformation to deal with
increasing variance of the response variable, that is, the
log-transformation, which could help us on this matter. Therefore, Table
3 again shows after the log-transformation of shares, the mean values
are similar to their corresponding median values, and their standard
deviations are much smaller than before relatively speaking.

Table 4 shows the numerical summaries of *average keywords* from bus
channel in mashable.com on different days of the week. This table
indicates the number of times *average keywords* shown in the articles
regarding the average number of shares, and the table is showing the
average number of those *average keywords* calculated for each day of
the week so that we can compare to see which day of the week, the
*average keywords* showed up the most or the worst according to the
average of shares in the bus channel.

Table 5 shows the numerical summaries of average shares of referenced
articles in mashable.com on different days of the week. We calculated
the average number of shares of those articles that contained the
earlier popularity of news referenced for each day of the week so that
we can compare which day has the most or the worst average number of
shares when there were earlier popularity of news referenced in the
busarticles.

Table 6 checks the numerical summaries of the `global_subjectivity`
variable between popular and unpopular articles, to see if there’s any
difference or a higher variation in subjectivity in popular articles.
Text subjectivity is a value between 0 and 1, so there isn’t any need
for transformation.

Table 7 checks the numerical summaries of the image count per article on
different days of the week, to see if there is a noticeable difference
in image count on weekends versus weekdays across all channels, or only
certain ones. Much like in table 2, the mean is smaller than the
standard deviation for most of the days of the week, and the solution
isn’t as straightforward, since many of the articles don’t have any
images at all. I’ll additionally include a log transformation of
`images + 1` to account for this.

``` r
# contingency table
edadata <- train
edadata$class.shares <- cut(edadata$class_shares, 2, c("Unpopular","Popular"))
edadata$day.week <- cut(edadata$dayweek, 7, c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
table(edadata$class.shares, edadata$day.week) %>% kable(caption = "Table 2. Popularity on Day of the Week")
```

|           | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday |
|:----------|-------:|--------:|----------:|---------:|-------:|---------:|-------:|
| Unpopular |    393 |     424 |       491 |      421 |    264 |       16 |     52 |
| Popular   |    428 |     401 |       418 |      430 |    312 |      149 |    183 |

Table 2. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 3. Average Shares vs. Average Log(shares) on Day of the Week")
```

| day.week  | Avg.shares | Sd.shares | Median.shares | Avg.logshares | Sd.logshares | Median.logshares |
|:----------|-----------:|----------:|--------------:|--------------:|-------------:|-----------------:|
| Monday    |   4191.832 | 33278.159 |          1400 |        7.4109 |       0.8955 |           7.2442 |
| Tuesday   |   2934.463 | 12066.613 |          1300 |        7.3526 |       0.8401 |           7.1701 |
| Wednesday |   2714.173 |  8858.365 |          1300 |        7.3091 |       0.8298 |           7.1701 |
| Thursday  |   2997.910 | 15204.307 |          1400 |        7.3471 |       0.7992 |           7.2442 |
| Friday    |   2467.287 |  5969.008 |          1450 |        7.3906 |       0.7835 |           7.2787 |
| Saturday  |   4951.697 | 12055.111 |          2700 |        8.0058 |       0.7943 |           7.9010 |
| Sunday    |   3500.425 |  5376.677 |          2100 |        7.7978 |       0.7243 |           7.6497 |

Table 3. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average Keywords on Day of the Week")
```

| day.week  | Avg.keyword | Sd.keyword | Median.keyword | IQR.keyword |
|:----------|------------:|-----------:|---------------:|------------:|
| Monday    |    2935.819 |   1861.488 |       2755.417 |   1066.8790 |
| Tuesday   |    2922.067 |   1114.094 |       2753.349 |    971.1080 |
| Wednesday |    2885.239 |   1070.991 |       2746.641 |    910.7842 |
| Thursday  |    2896.107 |   1245.945 |       2727.643 |   1004.7583 |
| Friday    |    2996.119 |   1969.579 |       2742.287 |    901.6783 |
| Saturday  |    3609.733 |   2763.920 |       3353.134 |   1187.8104 |
| Sunday    |    3193.893 |   1040.154 |       3129.464 |   1179.2366 |

Table 4. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 5. Summary of Average shares of referenced articles in Mashable on Day of the Week")
```

| day.week  | Avg.reference | Sd.reference | Median.reference | IQR.reference |
|:----------|--------------:|-------------:|-----------------:|--------------:|
| Monday    |      7003.652 |    34130.026 |         1950.000 |      3766.500 |
| Tuesday   |      5369.814 |    14573.626 |         2066.667 |      3230.000 |
| Wednesday |      7903.172 |    40798.364 |         2100.000 |      3789.000 |
| Thursday  |      5449.221 |    17830.870 |         2050.000 |      4100.000 |
| Friday    |      5637.258 |    17078.782 |         2009.071 |      3107.750 |
| Saturday  |      3550.522 |     8456.240 |         1300.000 |      3900.000 |
| Sunday    |      3113.325 |     5203.998 |         1510.667 |      3441.667 |

Table 5. Summary of Average shares of referenced articles in Mashable on
Day of the Week

``` r
edadata %>% group_by(class.shares) %>% summarize(
  Avg.subjectivity = mean(global_subjectivity), Sd.subjectivity = sd(global_subjectivity), 
  Median.subjectivity = median(global_subjectivity)) %>% 
  kable(digits = 4, caption = "Table 6. Comparing Global Subjectivity between Popular and Unpopular Articles")
```

| class.shares | Avg.subjectivity | Sd.subjectivity | Median.subjectivity |
|:-------------|-----------------:|----------------:|--------------------:|
| Unpopular    |           0.4268 |          0.0840 |              0.4281 |
| Popular      |           0.4447 |          0.0838 |              0.4498 |

Table 6. Comparing Global Subjectivity between Popular and Unpopular
Articles

``` r
edadata %>% group_by(day.week) %>% summarize(
  Avg.images = mean(num_imgs), Sd.images = sd(num_imgs), Median.images = median(num_imgs), Avg.log.images = mean(log(num_imgs + 1)), Sd.log.images = sd(log(num_imgs + 1)), Median.log.images = median(log(num_imgs + 1))) %>%
  kable(digits = 4, caption = "Table 7. Comparing Image Counts by the Day of the Week")
```

| day.week  | Avg.images | Sd.images | Median.images | Avg.log.images | Sd.log.images | Median.log.images |
|:----------|-----------:|----------:|--------------:|---------------:|--------------:|------------------:|
| Monday    |     1.8794 |    3.9185 |             1 |         0.7841 |        0.5778 |            0.6931 |
| Tuesday   |     1.8158 |    3.5890 |             1 |         0.7778 |        0.5676 |            0.6931 |
| Wednesday |     1.6458 |    3.0517 |             1 |         0.7629 |        0.5222 |            0.6931 |
| Thursday  |     1.8273 |    3.2398 |             1 |         0.7956 |        0.5694 |            0.6931 |
| Friday    |     1.8038 |    3.6596 |             1 |         0.7776 |        0.5573 |            0.6931 |
| Saturday  |     2.2545 |    4.9271 |             1 |         0.8832 |        0.6022 |            0.6931 |
| Sunday    |     2.1277 |    4.3344 |             1 |         0.8647 |        0.5951 |            0.6931 |

Table 7. Comparing Image Counts by the Day of the Week

## 3.2. Visualizations

Graphical presentation is a great tool used to visualize the
relationships between the predictors and the number of shares (or log
number of shares). Below we will see some plots that tell us stories
between those variables.

### 3.2.1. Correlation Plot

Figure 1 shows the linear relationship between the variables, both the
response and the predictors, which will be used in the regression models
as well as the ensemble models for predicting the number of shares.
Notice that there may be potential collinearity among the predictor
variables. The correlation ranges between -1 and 1, with the value
equals 0 means that there is no linear relationship between the two
variables. The closer the correlation measures towards 1, the stronger
the positive linear correlation/relationship there is between the two
variables. Vice verse, the close the correlation measures towards -1,
the stronger the negative linear correlation/relationship there is
between the two variables.

The correlation measures the “linear” relationships between the
variables. If the relationships between the variables are not linear,
then correlation measures cannot capture them, for instance, a quadratic
relationship. Scatterplots between the variables may be easier to spot
those non-linear relationships between the variables which we will show
in the following section.

``` r
correlation <- cor(train1, method="spearman")

corrplot(correlation, type = "upper", tl.pos = "lt", 
         title="Figure 1. Correlations Between the Variables",
         mar = c(0, 0, 2, 0))
corrplot(correlation, type = "lower", method = "number", add = TRUE, diag = FALSE, tl.pos = "n")
```

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/bus_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

### 3.2.2. Boxplot

Figure 2 shows the number of shares across different days of the week.
Here, due to the huge number of large-valued outliers, I capped the
number of shares to 8,000 so that we can see the medians and the
interquartile ranges clearly for different days of the week.

This is a boxplot with the days of the week on the x-axis and the number
of shares on the y-axis. We can inspect the trend of shares to see if
the shares are higher on a Monday, a Friday or a Sunday for the bus
articles.

Figure 2 coincides with the findings in Table 3 that the variance of
shares is huge across days of the week, and the mean values of shares
across different days are driven by larged-valued outliers. Therefore,
those mean values of shares are not close to the median values of shares
for each day of the week.

Figure 3 is a boxplot that compares the word count of the content of
each article, grouped by the day of the week and additionally colored by
its popularity. Similarly to before, for ease of viewing, the word count
is capped at 2,000 since a small number of articles have a much larger
word count in some cases. This has the capacity to indicate behavior
regarding article length on different days of the week, article
popularity based on length, or a cross-section of the two, where
articles of some length may be more or less popular on some days of the
week than others.

``` r
ggplot(data = edadata, aes(x = day.week, y = shares)) + 
  geom_boxplot(fill = "white", outlier.shape = NA) + 
  coord_cartesian(ylim=c(0, 8000)) + 
  geom_jitter(aes(color = day.week), size = 1) + 
  guides(color = guide_legend(override.aes = list(size = 8))) + 
  labs(x = "Day of the Week", y = "Number of Shares", 
       title = "Figure 2. Number of shares across different days of the week") + 
  scale_color_discrete(name = "Day of the Week") +
  theme(axis.text.x = element_text(angle = 45, size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 14))
```

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/bus_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
ggplot(data = edadata, aes(x = day.week, y = n_tokens_content)) +
  geom_boxplot(fill = "white", outlier.shape = NA) +
  coord_cartesian(ylim=c(0, 2000)) + 
  geom_jitter(aes(color = class.shares), size = 1) + 
  guides(color = guide_legend(override.aes = list(size = 8))) + 
  labs(x = "Popularity", y = "Word Count of Article", 
       title = "Figure 3. Word Count of Article by Popularity") + 
  theme(axis.text.x = element_text(angle = 45, size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 14))
```

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/bus_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

### 3.2.3.Barplot

Figure 4 shows the popularity of the news articles in relations to their
closeness to a top LDA topic for the bus channel on any day of the week.
The Latent Dirichlet Allocation (LDA) is an algorithm applied to the
Mashable texts of the articles in order to identify the five top
relevant topics and then measure the closeness of the current articles
to each topic, and there are five topics categories. Thus, each article
published on Mashable was measured for each of the topic categories.
Together, those LDA measures in ratios are added to 1 for each article.
Thus, these LDA topics variables are highly correlated with one another.

We calculated the mean ratios of these LDA topics variables for the
specific day of the week. These mean ratios are further classified into
a “popular” group and an “unpopular” group according to their number of
shares (&gt; 1400 or &lt; 1400) which is shown in Figure 4 barplot.
Note, the `position = "stack"` not `position = "fill"` in the `geom_bar`
function.

Some mean ratios of a LDA topic do not seem to vary over the days of a
week while other mean ratios of the LDA topics vary across different
days of the week. Recall, when we dichotomize a continuous variable into
different groups, we lose information about that variable. Here, I just
want to show you whether or not the mean ratios of a LDA topic differ
across time for different categories of shares.

``` r
b.plot1 <- edadata %>% group_by(day.week, class.shares) %>% 
  summarise(LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))

b.plot2 <- b.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")

ggplot(data = b.plot2, aes(x = day.week, y = avg.LDA, fill = LDA.Topic)) + 
  geom_bar(stat = "identity", position = "stack") + 
  labs(x = "Day of the Week", y = "Closeness to Top LDA Topic", 
       title = "Figure 4. Popularity of Top LDA Topic on Day of the Week") + 
  scale_fill_discrete(name = "LDA Topic") + 
  theme(axis.text.x = element_text(angle = 45, size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 13), 
        axis.title.y = element_text(size = 13), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13)) + 
  facet_wrap(~ class.shares)
```

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/bus_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

### 3.2.4. Line Plot

Figure 5 is a line plot that shows the same measurements as in Figure 4
that we can see the patterns of the mean ratios of a LDA topic vary or
not vary across time in different popularity groups more clearly. Again,
some mean ratios of LDA topics do not seem to vary across time when the
corresponding lines are flattened while other mean ratios of LDA topics
vary across time when their lines are fluctuating. The patterns observed
in the “popular” group may not reflect on the same trend in the
“unpopular” group for articles in the bus channel.

``` r
l.plot1 <- edadata %>% group_by(day.week, class.shares) %>% 
  summarise(LDA_0=mean(LDA_00), LDA_1=mean(LDA_01), LDA_2=mean(LDA_02), LDA_3=mean(LDA_03), LDA_4=mean(LDA_04))

l.plot2 <- l.plot1 %>% pivot_longer(cols = 3:7, names_to = "LDA.Topic", values_to = "avg.LDA")

ggplot(data = l.plot2, aes(x = day.week, y = avg.LDA, group = LDA.Topic)) + 
  geom_line(aes(color = LDA.Topic), lwd = 2) + 
  labs(x = "Day of the Week", y = "Closeness to LDA Topic", 
       title = "Figure 5. Popularity of LDA Topic on Day of the Week") + 
  scale_color_discrete(name = "LDA Topic") +
  theme(axis.text.x = element_text(angle = 45, size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13)) +
  facet_wrap(~ class.shares)
```

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/bus_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

### 3.2.5. Scatterplots

Figure 6 shows the relationship between the average keyword and
log-transformed shares for articles in the bus channel across different
days of the week. In the news popularity study, it showed average
keyword was ranked top one predictor in variable importance in the
optimal predictive model (random forest) they selected that produced the
highest accuracy in prediction of popularity online articles. Therefore,
we are interested to see how average keyword is related with log shares.
The different colored linear regression lines indicate different days of
the week.

If the points display an upward trend, it indicates a positive
relationship between the average keyword and log-shares. With an
increasing log number of shares, the number of average keywords also
increases, meaning people tend to share the article more when they see
more of those average keywords in the article. On the contrary, if the
points are in a downward trend, it indicates a negative relationship
between the average keyword and log-shares. With an decreasing log
number of shares, the number of average keywords decreases as well.
People tend to share the articles less when they see less of these
average keywords in the articles from the bus channel.

Figure 7 is similar, except it compares the log-transformed number of
shares to the log-transformed images in the article. As noted
previously, both of these variables do not behave properly in a linear
model due to the existence of extreme outliers in the data. Here, a
negative correlation will indicate that shares sharply decreased for
articles containing more images, and a positive correlation will
indicate that shares sharply increased for articles containing more
images.

``` r
ggplot(data = edadata, aes(x = kw_avg_avg, y = log.shares, color = day.week)) + 
  geom_point(size = 2) + #aes(shape = class.shares)
  scale_color_discrete(name = "Day of the Week") + 
  coord_cartesian(xlim=c(0, 10000)) +
  geom_smooth(method = "lm", lwd = 2) + 
  labs(x = "Average Keywords", y = "log(number of shares)", 
       title = "Figure 6. Average Keywords vs Log Number of Shares") + 
  theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/bus_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
ggplot(data = edadata, aes(x = log(num_imgs + 1), y = log.shares, color = day.week)) + 
  geom_point(size = 2) +
  scale_color_discrete(name = "Day of the Week") + 
  geom_smooth(method = "lm", lwd = 2) + 
  labs(x = "log(number of images)", y = "log(number of shares)", 
       title = "Figure 7. Log Number of Images vs Log Number of Shares") + 
  theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/bus_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

### 3.2.6. QQ Plots

To justify the usage of the log transformations for shares and images,
we’ll show the QQ plot of each over the bus channel in figures 8a, 8b,
8c, and 8d. We’re aiming for something close to a straight line, which
would indicate that the data is approximately normal in its distribution
and does not need further standardization.

``` r
ggplot(edadata) + geom_qq(aes(sample = shares)) + geom_qq_line(aes(sample = shares)) + 
  labs(x = "Theoretical Quantiles", y = "Share Numbers", 
       title = "Figure 8a. QQ Plot for Non-Transformed Shares") +
    theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/bus_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
ggplot(edadata) + geom_qq(aes(sample = log(shares))) + geom_qq_line(aes(sample = log(shares))) +
    labs(x = "Theoretical Quantiles", y = "Log(Share Numbers)", 
       title = "Figure 8b. QQ Plot for Log-Transformed Shares") +
    theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/bus_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
ggplot(edadata) + geom_qq(aes(sample = num_imgs)) + geom_qq_line(aes(sample = num_imgs)) + 
  labs(x = "Theoretical Quantiles", y = "Image Numbers", 
       title = "Figure 8c. QQ Plot for Non-Transformed Image Numbers") +
    theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/bus_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

``` r
ggplot(edadata) + geom_qq(aes(sample = log(num_imgs + 1))) + geom_qq_line(aes(sample = log(num_imgs + 1))) +
    labs(x = "Theoretical Quantiles", y = "Log(Image Numbers)", 
       title = "Figure 8d. QQ Plot for Log-Transformed Image Numbers") +
    theme(axis.text.x = element_text(size = 10), 
        axis.text.y = element_text(size = 10), 
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), 
        legend.key.size = unit(1, 'cm'), 
        legend.text = element_text(size = 13), 
        title = element_text(size = 13))
```

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/bus_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

Whether it’s appropriate to perform a logarithmic transformation on the
number of images is somewhat less clear than for the number of shares.

# 4. Candidate Models

## 4.1. Linear Regression

The linear regression process takes a matrix of all of the predictor
variables we’ve chosen and compares their values to each of the
corresponding values of the response variable, `log.shares`. This allows
us to calculate the most accurate linear combination of the predictor
variables to make up the response variable. We can choose a variety of
sets of predictors and compare their Root Mean Square Errors, R-Squared
values, and Mean Absolute Errors to see which one is the strongest
model. Below, we’ve fit multiple linear models that include all of our
variables and various combinations of interaction terms and/or quadratic
terms.

Originally, we used 20 top ranked predictors selected by the optimal
predictive model used in the paper and fit them in three types of models
below.

1.  All predictors  
2.  All predictors and their first order interactions  
3.  All predictors, their first order interactions and the quadratic
    terms

We slowly filtered out predictors, their corresponding second order
terms and the interaction terms that were insignificant at 0.05 level.
In addition, we also examined the predictors using correlation plots. We
slowly got rid of some predictors that are highly correlated with each
other such as the LDA topics variables which are also shown in Figures 3
and 4, average number of shares of keywords, maximum number of shares of
average keywords, minimum number of shares of average keywords and many
of others. We carefully monitored this process and compared the models
with the adjusted R-squared and RMSE values. Due to multi-collinearity
among the predictors, reducing the number of predictors that are
correlated with one another did not make the model fit worse, and the
RMSE value from the model was surprisingly decreased.

We repeated this process to trim down the number of predictors and
eventually selected the ones that seem to be important in predicting the
number of shares, and they are not highly correlated with each other.
The parameters in the linear regression model 1 were chosen through this
process, and they are listed in Table 8 below. The response variable is
`log(shares)`.

The second linear regression model provided below applies a log
transformation to the variables `num_imgs` and
`self_reference_avg_sharess`, which are extremely skewed to the right
even after standardization, to see if this accounts for the outliers in
the data and puts those variables in an appropriate context.

Table 8. The predictors in linear regression model 1

| Index | Predictor in Linear Regression 1           | Predictor Information                                               |
|-------|--------------------------------------------|---------------------------------------------------------------------|
| 1     | `kw_avg_avg`                               | Average keyword (average shares)                                    |
| 2     | `LDA_02`                                   | Closeness of current article to a LDA Topic 2                       |
| 3     | `dayweek`                                  | Day of a week the article was published                             |
| 4     | `self_reference_avg_sharess`               | Average shares of earlier popularity news referenced in the article |
| 5     | `average_token_length`                     | Average length of the words in the content                          |
| 6     | `n_tokens_content`                         | Number of words in the content                                      |
| 7     | `n_tokens_title`                           | Number of words in the title                                        |
| 8     | `global_subjectivity`                      | Text subjectivity                                                   |
| 9     | `num_imgs`                                 | Number of images                                                    |
| 10    | `I(n_tokens_content^2)`                    | Quadratic term of number of words in the content                    |
| 11    | `kw_avg_avg:num_imgs`                      | Interaction term                                                    |
| 12    | `average_token_length:global_subjectivity` | Interaction term                                                    |
| 13    | `dayweek:self_reference_avg_sharess`       | Interaction term                                                    |

We standardized both the training set and the test set before we fit the
linear regression models. We used the standardized mean and standard
deviation values from the training set on the test set so that the
predictors on both sets have the same means and standard deviations.
Standardizing the variables is very important prior to model fitting so
that all the variables are on the same scale. The variables with large
values do not look more important than the variables with smaller
values. Especially in our data set where some variable values are
unbounded, skewed to one side, and some variables are ratios between 0
and 1.

``` r
#train1$dayweek <- as.factor(train1$dayweek)
#test1$dayweek <- as.factor(test1$dayweek)
train1$dayweek <- cut(train1$dayweek, 7, c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
test1$dayweek <- cut(test1$dayweek, 7, c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
preProcValues <- preProcess(train1, method = c("center", "scale"))
trainTransformed1 <- predict(preProcValues, train1)
testTransformed1 <- predict(preProcValues, test1)
trainTransformed1
```

    ## # A tibble: 4,382 x 10
    ##    log.shares dayweek   kw_avg_avg LDA_02 self_reference_av~ average_token_le~ n_tokens_content n_tokens_title global_subjecti~
    ##         <dbl> <fct>          <dbl>  <dbl>              <dbl>             <dbl>            <dbl>          <dbl>            <dbl>
    ##  1     -1.00  Monday         -1.96 -0.284           -0.228              0.560            -0.640         -0.595          -1.13  
    ##  2      0.752 Monday         -1.96 -0.432           -0.228              1.88             -0.317         -1.06           -0.734 
    ##  3     -0.784 Monday         -1.96 -0.285           -0.123             -0.656            -0.665          1.26           -1.24  
    ##  4      0.790 Monday         -1.96 -0.474            0.00141           -0.167             0.391         -1.06            0.654 
    ##  5     -1.25  Monday         -1.96  1.38            -0.228             -1.02             -0.897         -0.132           0.0781
    ##  6     -0.831 Monday         -1.96 -0.549           -0.190              0.305            -0.210          0.794           0.307 
    ##  7      0.231 Wednesday      -1.43 -0.285           -0.153             -0.129            -0.690         -0.595          -3.00  
    ##  8      0.170 Wednesday      -1.53 -0.373           -0.183             -0.390            -0.156         -0.132           0.0238
    ##  9      0.170 Wednesday      -1.32 -0.549           -0.149              0.616            -0.827          0.331           2.20  
    ## 10     -1.11  Wednesday      -1.34 -0.474           -0.0512            -0.0740           -0.470         -0.595          -0.964 
    ## # ... with 4,372 more rows, and 1 more variable: num_imgs <dbl>

``` r
cv_fit1 <- train(log.shares ~ . + I(n_tokens_content^2) + kw_avg_avg:num_imgs + 
                   average_token_length:global_subjectivity + dayweek:self_reference_avg_sharess, 
                 data=trainTransformed1,
                 method = "lm",
                 trControl = trainControl(method = "cv", number = 10))
summary(cv_fit1)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -9.2612 -0.5885 -0.1580  0.4267  6.8589 
    ## 
    ## Coefficients:
    ##                                                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                    0.004478   0.033573   0.133 0.893902    
    ## dayweekTuesday                                -0.052591   0.046542  -1.130 0.258555    
    ## dayweekWednesday                              -0.096230   0.045455  -2.117 0.034311 *  
    ## dayweekThursday                               -0.048093   0.046200  -1.041 0.297944    
    ## dayweekFriday                                 -0.023356   0.051303  -0.455 0.648952    
    ## dayweekSaturday                                0.530576   0.084413   6.286 3.59e-10 ***
    ## dayweekSunday                                  0.412706   0.078942   5.228 1.79e-07 ***
    ## kw_avg_avg                                     0.169528   0.016150  10.497  < 2e-16 ***
    ## LDA_02                                        -0.017877   0.014389  -1.242 0.214148    
    ## self_reference_avg_sharess                     0.099611   0.026361   3.779 0.000160 ***
    ## average_token_length                          -0.078146   0.022114  -3.534 0.000414 ***
    ## n_tokens_content                               0.139419   0.020887   6.675 2.79e-11 ***
    ## n_tokens_title                                 0.001052   0.014459   0.073 0.941987    
    ## global_subjectivity                            0.088874   0.015414   5.766 8.69e-09 ***
    ## num_imgs                                       0.037429   0.014787   2.531 0.011404 *  
    ## `I(n_tokens_content^2)`                       -0.003346   0.005182  -0.646 0.518533    
    ## `kw_avg_avg:num_imgs`                          0.069074   0.019353   3.569 0.000362 ***
    ## `average_token_length:global_subjectivity`    -0.001760   0.005834  -0.302 0.762941    
    ## `dayweekTuesday:self_reference_avg_sharess`    0.041491   0.065293   0.635 0.525160    
    ## `dayweekWednesday:self_reference_avg_sharess` -0.075615   0.033125  -2.283 0.022496 *  
    ## `dayweekThursday:self_reference_avg_sharess`   0.049243   0.054732   0.900 0.368323    
    ## `dayweekFriday:self_reference_avg_sharess`    -0.153395   0.068143  -2.251 0.024430 *  
    ## `dayweekSaturday:self_reference_avg_sharess`  -0.099929   0.233289  -0.428 0.668417    
    ## `dayweekSunday:self_reference_avg_sharess`     0.247083   0.317006   0.779 0.435770    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9425 on 4358 degrees of freedom
    ## Multiple R-squared:  0.1163, Adjusted R-squared:  0.1116 
    ## F-statistic: 24.93 on 23 and 4358 DF,  p-value: < 2.2e-16

``` r
cv_fit2 <- train(log.shares ~ . - num_imgs - self_reference_avg_sharess + I(log(num_imgs+1)) + I(n_tokens_content^2) +
                 I(log(self_reference_avg_sharess+1)) + kw_avg_avg:I(log(num_imgs + 1)) +
                 average_token_length:global_subjectivity, 
                 data=trainTransformed1,
                 method = "lm",
                 trControl = trainControl(method = "cv", number = 10))
summary(cv_fit2)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -9.1990 -0.5850 -0.1496  0.4305  6.9211 
    ## 
    ## Coefficients:
    ##                                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.0568407  0.0339967   1.672 0.094606 .  
    ## dayweekTuesday                             -0.0558133  0.0462517  -1.207 0.227602    
    ## dayweekWednesday                           -0.1023745  0.0451725  -2.266 0.023482 *  
    ## dayweekThursday                            -0.0503112  0.0459299  -1.095 0.273406    
    ## dayweekFriday                              -0.0209764  0.0509898  -0.411 0.680811    
    ## dayweekSaturday                             0.5432095  0.0811275   6.696 2.42e-11 ***
    ## dayweekSunday                               0.3912867  0.0706918   5.535 3.29e-08 ***
    ## kw_avg_avg                                  0.1818014  0.0194046   9.369  < 2e-16 ***
    ## LDA_02                                     -0.0186425  0.0143073  -1.303 0.192643    
    ## average_token_length                       -0.0731875  0.0219683  -3.332 0.000871 ***
    ## n_tokens_content                            0.1363888  0.0211364   6.453 1.22e-10 ***
    ## n_tokens_title                              0.0022600  0.0143670   0.157 0.875014    
    ## global_subjectivity                         0.0850809  0.0153425   5.545 3.11e-08 ***
    ## `I(log(num_imgs + 1))`                      0.0975682  0.0331707   2.941 0.003285 ** 
    ## `I(n_tokens_content^2)`                    -0.0028634  0.0051538  -0.556 0.578525    
    ## `I(log(self_reference_avg_sharess + 1))`    0.4147480  0.0500215   8.291  < 2e-16 ***
    ## `kw_avg_avg:I(log(num_imgs + 1))`           0.1166217  0.0322124   3.620 0.000297 ***
    ## `average_token_length:global_subjectivity`  0.0003554  0.0057953   0.061 0.951098    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9373 on 4364 degrees of freedom
    ## Multiple R-squared:  0.1248, Adjusted R-squared:  0.1214 
    ## F-statistic: 36.61 on 17 and 4364 DF,  p-value: < 2.2e-16

``` r
pred1 <- predict(cv_fit1, newdata = testTransformed1)
pred2 <- predict(cv_fit2, newdata = testTransformed1)
cv_rmse1 <- postResample(pred1, obs = testTransformed1$log.shares)
cv_rmse2 <- postResample(pred2, obs = testTransformed1$log.shares)
```

## 4.2.Random Forest

The bootstrap approach to fitting a tree model involves resampling our
data and fitting a tree to each sample, and then averaging the resulting
predictions of each of those models. The random forest approach adds an
extra step for each of these samples, where only a random subset of the
predictor variables is chosen each time, in order to reduce the
correlation between each of the trees. We don’t have to worry about
creating dummy variables for categorical variables, because our data
already comes in an entirely numeric form.

Again, we standardized both the training set and the test set before we
fit the each of the ensemble models. We used the standardized mean and
standard deviation values of the predictors from the training set on the
test set so that the predictors on both sets have the same means and
standard deviations.

``` r
train2 <- train %>% select(-class_shares, -shares, -dayweek, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
test2 <- test %>% select(-class_shares, -shares, -dayweek, -LDA_00, -LDA_01, -LDA_03, -LDA_04)
preProcValues2 <- preProcess(train2, method = c("center", "scale"))
trainTransformed2 <- predict(preProcValues2, train2)
testTransformed2 <- predict(preProcValues2, test2)
trainTransformed2
```

    ## # A tibble: 4,382 x 16
    ##    log.shares kw_avg_avg LDA_02 weekday_is_monday weekday_is_tuesday weekday_is_wednesday weekday_is_thursday weekday_is_friday
    ##         <dbl>      <dbl>  <dbl>             <dbl>              <dbl>                <dbl>               <dbl>             <dbl>
    ##  1     -1.00       -1.96 -0.284             2.08              -0.482               -0.512              -0.491            -0.389
    ##  2      0.752      -1.96 -0.432             2.08              -0.482               -0.512              -0.491            -0.389
    ##  3     -0.784      -1.96 -0.285             2.08              -0.482               -0.512              -0.491            -0.389
    ##  4      0.790      -1.96 -0.474             2.08              -0.482               -0.512              -0.491            -0.389
    ##  5     -1.25       -1.96  1.38              2.08              -0.482               -0.512              -0.491            -0.389
    ##  6     -0.831      -1.96 -0.549             2.08              -0.482               -0.512              -0.491            -0.389
    ##  7      0.231      -1.43 -0.285            -0.480             -0.482                1.95               -0.491            -0.389
    ##  8      0.170      -1.53 -0.373            -0.480             -0.482                1.95               -0.491            -0.389
    ##  9      0.170      -1.32 -0.549            -0.480             -0.482                1.95               -0.491            -0.389
    ## 10     -1.11       -1.34 -0.474            -0.480             -0.482                1.95               -0.491            -0.389
    ## # ... with 4,372 more rows, and 8 more variables: weekday_is_saturday <dbl>, weekday_is_sunday <dbl>,
    ## #   self_reference_avg_sharess <dbl>, average_token_length <dbl>, n_tokens_content <dbl>, n_tokens_title <dbl>,
    ## #   global_subjectivity <dbl>, num_imgs <dbl>

``` r
random_forest <- train(log.shares ~ ., data = trainTransformed2,
    method = "rf",
    trControl = trainControl(method = "cv", number = 5),
    tuneGrid = data.frame(mtry = 1:5), importance = TRUE)

random_forest_predict <- predict(random_forest, newdata = testTransformed2)
rf_rmse <- postResample(random_forest_predict, obs = testTransformed2$log.shares)
random_forest
```

    ## Random Forest 
    ## 
    ## 4382 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 3506, 3506, 3505, 3505, 3506 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE       Rsquared   MAE      
    ##   1     0.9411067  0.1563596  0.6835409
    ##   2     0.9158902  0.1638697  0.6584750
    ##   3     0.9162359  0.1600127  0.6581749
    ##   4     0.9194078  0.1548744  0.6598793
    ##   5     0.9210318  0.1528424  0.6605471
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 2.

``` r
random_forest$results
```

    ## # A tibble: 5 x 7
    ##    mtry  RMSE Rsquared   MAE RMSESD RsquaredSD  MAESD
    ##   <int> <dbl>    <dbl> <dbl>  <dbl>      <dbl>  <dbl>
    ## 1     1 0.941    0.156 0.684 0.0523     0.0256 0.0180
    ## 2     2 0.916    0.164 0.658 0.0528     0.0292 0.0217
    ## 3     3 0.916    0.160 0.658 0.0521     0.0304 0.0219
    ## 4     4 0.919    0.155 0.660 0.0527     0.0298 0.0208
    ## 5     5 0.921    0.153 0.661 0.0497     0.0271 0.0198

``` r
# mtry = random_forest$bestTune[[1]]
```

We then used 5 fold cross validation to search for the tuning parameter
value ranging from 1 to 5 that produces the optimal random forest model.
The optimal model chosen by cross validation produced the smallest RMSE
value when mtry = 2 and the lowest RMSE = 0.9158902 when training the
model on the training set.

## 4.3. Boosted Tree

Random forest models use bagging technique (bootstrap aggregation) to
build independent decision trees with different subsets of predictors
and combine them in parallel. On the contrary, gradient boosted trees
use a method called boosting. Boosting method trains each weak learner
slowly and then combines them sequentially, with weak learners being the
decision trees with only one split. Thus, each new tree can correct the
errors made by the previous tree. Because boosting is to slowly train
the trees so that they avoid overfitting the data. Since the trees grow
slowly and in a sequential manner, each tree we create is based off a
previous tree and we update the predictions as we go. For instance, we
fit the model, we get our predictions, and now we create a new model
based off the previous model. Then, we update our predictions based on
the new model. Then, we build a newer model based off the previous one,
and we update the predictions from the new model. The process is
repeated until the criteria set for the tuning parameters are met.

There are tuning parameters in gradient boosting machine learning
technique to help us prevent from growing the trees too quickly and thus
keep us from overfitting the model.

-   `shrinkage`: A shrinkage parameter controls the growth rate of the
    trees, slows fitting process  
-   `n.trees`: The amount of times we want to the process to repeat in
    training the trees.  
-   `interaction.depth`: The amount of splits we want to fit a tree.  
-   `n.minobsinnode`: The minimum number of observations in a node at
    least.

Here, we use the `caret` package and the `gbm` package to run the
boosted tree model with the training set and predict on the test set.
The values of the tuning parameters are set as below:

-   `shrinkage` = 0.1  
-   `n.trees` = 25, 50, 75, 100, 125  
-   `interaction.depth` = 1, 2, 3, 4, 5  
-   `n.minobsinnode` = 10

We then used 10 fold cross validation to search all combinations of the
tuning parameters values using the `expand.grid` function to choose an
optimal model with the desired tuning parameters values. The optimal
model chosen by cross validation across all combinations of tuning
parameters values produces the lowest root mean squared error (RMSE).

``` r
#expand.grid(n.trees = c(25, 50, 75, 100, 125), interaction.depth = 1:5, shrinkage = 0.1, n.minobsinnode = 10)
boosted_tree <- train(log.shares ~ . , data = trainTransformed2,
      method = "gbm", 
      trControl = trainControl(method = "cv", number = 10), #method="repeatedcv", repeats=5
      tuneGrid = expand.grid(n.trees = c(25, 50, 75, 100, 125), interaction.depth = 1:5, shrinkage = 0.1, n.minobsinnode = 10),
      verbose = FALSE)
boosted_tree_predict <- predict(boosted_tree, newdata = select(testTransformed2, -log.shares))
boost_rmse <- postResample(boosted_tree_predict, obs = testTransformed2$log.shares)
boosted_tree
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 4382 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 3943, 3944, 3945, 3943, 3944, 3944, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared   MAE      
    ##   1                   25      0.9413864  0.1245344  0.6848905
    ##   1                   50      0.9284952  0.1395063  0.6726734
    ##   1                   75      0.9240500  0.1444586  0.6680301
    ##   1                  100      0.9215064  0.1483318  0.6665020
    ##   1                  125      0.9208405  0.1489224  0.6655325
    ##   2                   25      0.9327866  0.1343654  0.6754778
    ##   2                   50      0.9229082  0.1462465  0.6656211
    ##   2                   75      0.9221525  0.1473731  0.6644051
    ##   2                  100      0.9217861  0.1485082  0.6636602
    ##   2                  125      0.9229388  0.1474210  0.6642765
    ##   3                   25      0.9274804  0.1418958  0.6703679
    ##   3                   50      0.9224699  0.1468606  0.6634021
    ##   3                   75      0.9201389  0.1523150  0.6609479
    ##   3                  100      0.9189540  0.1546077  0.6606897
    ##   3                  125      0.9193825  0.1542767  0.6601443
    ##   4                   25      0.9257557  0.1435243  0.6674832
    ##   4                   50      0.9205741  0.1504303  0.6616667
    ##   4                   75      0.9202474  0.1521945  0.6604379
    ##   4                  100      0.9222274  0.1501238  0.6607886
    ##   4                  125      0.9231547  0.1492236  0.6613711
    ##   5                   25      0.9227460  0.1487691  0.6666369
    ##   5                   50      0.9205725  0.1508236  0.6611568
    ##   5                   75      0.9203363  0.1528677  0.6606067
    ##   5                  100      0.9223799  0.1498959  0.6612552
    ##   5                  125      0.9244609  0.1476367  0.6619311
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at
    ##  a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 100, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
#boosted_tree$results
# n.trees = boosted_tree$bestTune[[1]]
```

We then used 10 fold cross validation to search all combinations of the
tuning parameters values using the `expand.grid` function to choose an
optimal model with the desired tuning parameters values. The optimal
boosted tree model chosen by cross validation produced the smallest RMSE
value (0.918954) when n.trees = 100, interaction.depth = 3, shrinkage =
0.1 and n.minobsinnode = 10 when training the model with the training
set.

## 4.4. Model Comparisons

The best model fit to predict the number of shares for the bus channel
can be determined by looking at the Root Mean Squared Error, the Mean
Absolute Error, or the R-squared value using the test set. Table 8 shows
these criteria measures for each candidate model. The approach I’ve
taken below picks whichever model is considered superior by the majority
score of these three metrics, and if each of the metrics picks a
different model or the majority score is a tiebreaker, then the one with
the lowest RMSE will be chosen.

``` r
result2 <- rbind(cv_rmse1, cv_rmse1, rf_rmse, boost_rmse)
row.names(result2) <- c("Linear Regression 1", "Linear Regression 2", "Random Forest", "Boosted Tree")
kable(result2, digits = 4, caption = "Table 8. Cross Validation - Comparisons of the models in test set")
```

|                     |   RMSE | Rsquared |    MAE |
|:--------------------|-------:|---------:|-------:|
| Linear Regression 1 | 0.9305 |   0.1267 | 0.6878 |
| Linear Regression 2 | 0.9305 |   0.1267 | 0.6878 |
| Random Forest       | 0.8954 |   0.1997 | 0.6593 |
| Boosted Tree        | 0.9030 |   0.1772 | 0.6655 |

Table 8. Cross Validation - Comparisons of the models in test set

``` r
rmse_best <- names(which.min(result2[,1]))
rsq_best <- names(which.max(result2[,2]))
mae_best <- names(which.min(result2[,3]))
model_best <- table(c(rmse_best, rsq_best, mae_best))
final_model <- if_else(max(model_best) > 1, names(which.max(model_best)), rmse_best)
```

We built a helper function so that when a final winning model is
declared among the candidate models using the test set, we then used the
final model to fit the entire data set, both the training and the test
sets, from the bus channel. A variable importance plot is produced along
with a table containing a ranking metrics of the variable importance
when fitting the final model with the entire data set of the bus
articles.

``` r
linear1 <- function(...){
  
  data1comb <- rbind(trainTransformed1, testTransformed1, ...)
  linearfit1 <- train(log.shares ~ . + I(n_tokens_content^2) + kw_avg_avg:num_imgs + 
                        average_token_length:global_subjectivity + dayweek:self_reference_avg_sharess, 
                      data=data1comb, 
                      method = "lm", ...)
  pred <- predict(linearfit1, newdata = data1comb, ...)
  linear.rmse <- postResample(pred, obs = data1comb$log.shares, ...)
  #colnames(linear.rmse) <- "Linear Regression 1"
  linear_rmse <- kable(linear.rmse, digits = 4, caption = "10. Final model fit on entire data", ...)
  summary <- summary(linearfit1)
  important <- varImp(linearfit1, ...)
  plot_imp <- vip(linearfit1, ...)
  return(list(linear_rmse, summary, important, plot_imp))
  
}

linear2 <- function(...){
  
  data1comb <- rbind(trainTransformed1, testTransformed1, ...)
  linearfit2 <- train(log.shares ~ . - num_imgs - self_reference_avg_sharess + I(n_tokens_content^2) + 
                        I(log(num_imgs + 1)) + I(log(self_reference_avg_sharess+1)) + 
                        kw_avg_avg:I(log(num_imgs + 1)) + average_token_length:global_subjectivity, 
                      data=data1comb, 
                      method = "lm", ...)
  pred <- predict(linearfit2, newdata = data1comb, ...)
  linear.rmse <- postResample(pred, obs = data1comb$log.shares, ...)
  #row.names(linear.rmse) <- "Linear Regression 2"
  linear_rmse <- kable(linear.rmse, digits = 4, caption = "10. Final model fit on entire data", ...)
  summary <- summary(linearfit2)
  return(list(linear_rmse, summary, important, plot_imp))
  
}

randomf <- function(...){
  
  data2comb <- rbind(trainTransformed2, testTransformed2, ...)
  rffit <- train(log.shares ~ . , data = data2comb, 
                 method = "rf", 
                 tuneGrid = data.frame(mtry = random_forest$bestTune[[1]]), 
                 importance = TRUE, ...)
  important <- varImp(rffit , ...)
  plot_imp <- vip(rffit, ...)
  return(list(rffit, important, plot_imp))
  
}

boostedt <- function(...){
  
  data2comb <- rbind(trainTransformed2, testTransformed2, ...)
  btfit <- train(log.shares ~ . , data = data2comb, 
                 method = "gbm", 
                 tuneGrid = data.frame(n.trees = boosted_tree$bestTune[[1]], 
                                       interaction.depth = boosted_tree$bestTune[[2]], 
                                       shrinkage = boosted_tree$bestTune[[3]], 
                                       n.minobsinnode = boosted_tree$bestTune[[4]]), 
                 verbose = FALSE, ...)
  important <- varImp(btfit, ...)
  plot_imp <- vip(btfit, ...)
  return(list(btfit, important, plot_imp))
  
}
```

# 5. Final Model Fit with Entire Data

The best model fit to predict the number of shares is the **Random
Forest** model for the bus articles. We fit the entire data set, both
the training and the test set from the channel in the final chosen
model. The values of RMSE, MAE and R-squared are calculated with the
entire data set using the final model. A variable importance plot of the
top 10 most important variables and a table containing a ranking metric
of the relative variable importance are produced below. We can examine
which predictors contributed the most in predicting the popularity of
online news in the final model accordingly. Table 1 containing attribute
information from the *Introduction* section is copied below for
comparisons of variable importance.

Table 1. Attributes used in the analyses for prediction of online news
popularity

| Index | Attribute                    | Attribute Information                                  | Type    |
|-------|------------------------------|--------------------------------------------------------|---------|
| 1     | `shares` (target)            | Number of shares                                       | number  |
| 2     | `kw_avg_avg`                 | Average keyword (average shares)                       | number  |
| 3     | `LDA_02`                     | Closeness of current article to a LDA Topic 2          | ratio   |
| 4.1   | `weekday_is_monday`          | Was the article published on a Monday?                 | boolean |
| 4.2   | `weekday_is_tuesday`         | Was the article published on a Tuesday?                | boolean |
| 4.3   | `weekday_is_wednesday`       | Was the article published on a Wednesday?              | boolean |
| 4.4   | `weekday_is_thursday`        | Was the article published on a Thursday?               | boolean |
| 4.5   | `weekday_is_friday`          | Was the article published on a Friday?                 | boolean |
| 4.6   | `weekday_is_saturday`        | Was the article published on a Saturday?               | boolean |
| 4.7   | `weekday_is_sunday`          | Was the article published on a Sunday?                 | boolean |
| 5     | `self_reference_avg_sharess` | Avg. shares of popular news referenced in the articles | number  |
| 6     | `average_token_length`       | Average length of the words in the content             | number  |
| 7     | `n_tokens_content`           | Number of words in the content                         | number  |
| 8     | `n_tokens_title`             | Number of words in the title                           | number  |
| 9     | `global_subjectivity`        | Text subjectivity                                      | ratio   |
| 10    | `num_imgs`                   | Number of images                                       | number  |

``` r
f_model <- if_else(final_model == "Random Forest", 3, 
           if_else(final_model == "Boosted Tree", 4, 
                   if_else(final_model == "Linear Regression 1", 1, 2)))

switch(f_model,
       
       modell = linear1(),
       model2 = linear2(),
       model3 = randomf(),
       model4 = boostedt()
)
```

    ## [[1]]
    ## Random Forest 
    ## 
    ## 6258 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 6258, 6258, 6258, 6258, 6258, 6258, ... 
    ## Resampling results:
    ## 
    ##   RMSE       Rsquared   MAE      
    ##   0.9139561  0.1707004  0.6600953
    ## 
    ## Tuning parameter 'mtry' was held constant at a value of 2
    ## 
    ## [[2]]
    ## rf variable importance
    ## 
    ##                            Overall
    ## kw_avg_avg                 100.000
    ## weekday_is_saturday         80.238
    ## n_tokens_content            77.675
    ## weekday_is_sunday           75.416
    ## self_reference_avg_sharess  51.573
    ## num_imgs                    35.551
    ## LDA_02                      30.783
    ## average_token_length        29.808
    ## global_subjectivity         25.670
    ## weekday_is_wednesday        17.967
    ## weekday_is_thursday         14.598
    ## weekday_is_friday           14.297
    ## weekday_is_monday            8.942
    ## n_tokens_title               7.576
    ## weekday_is_tuesday           0.000
    ## 
    ## [[3]]

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/bus_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

``` r
#imp <- varImp(boosted_tree)
#imp2 <- data.frame(imp[1])
#first <- rownames(imp2)[order(imp2$Overall, decreasing=TRUE)][1] # most important
#second <- rownames(imp2)[order(imp2$Overall, decreasing=TRUE)][2] #2nd
#third <- rownames(imp2)[order(imp2$Overall, decreasing=TRUE)][3] # 3rd
#print(paste0("The most important predictor is ", first, ", the second is ", second))
```

# 6. Automation

To knit with the parameters with this single .Rmd file, we specified the
parameters in the YAML header and created a data frame which contains
both the output .md file names and the parameters we supplied to the
YAML header to automate the reports. Both the names of the parameters
and the values of the parameters were saved in a list format in that
data frame. Then, we used the render function below to knit with the
parameters in R console to generate the reports. The render function is
not evaluated with this .Rmd file.

``` r
type <- c("lifestyle", "entertainment", "bus", "socmed", "tech", "world")
output_file <- paste0(type, ".md")
params <- lapply(type, FUN = function(x){list(channel = x)})
reports <- tibble(output_file, params)

apply(reports, MARGIN = 1, 
      FUN = function(x){
        render(input = "C:/Users/peach/Documents/ST558/ST558_repos/News-Popularity-Prediction/ST558_project2_auto.Rmd",
               output_format = "github_document", 
               output_file = paste0("C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/", x[[1]]),
               params = x[[2]],
               output_options = list(html_preview = FALSE, toc = TRUE, toc_depth = 3, df_print = "tibble"))
      })
```
