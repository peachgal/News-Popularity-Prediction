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

    ## # A tibble: 5,145 x 23
    ##    class_shares shares log.shares dayweek kw_avg_avg LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 weekday_is_monday weekday_is_tuesday
    ##           <dbl>  <dbl>      <dbl>   <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>             <dbl>              <dbl>
    ##  1            0    505       6.22       1          0 0.0286 0.0288 0.0286 0.0286  0.885                 1                  0
    ##  2            0    855       6.75       1          0 0.0222 0.307  0.0222 0.0222  0.627                 1                  0
    ##  3            1  17100       9.75       1          0 0.0250 0.0252 0.0250 0.0250  0.900                 1                  0
    ##  4            1   2800       7.94       1          0 0.0201 0.0200 0.0200 0.0200  0.920                 1                  0
    ##  5            0    445       6.10       1          0 0.312  0.233  0.0286 0.0286  0.398                 1                  0
    ##  6            0    783       6.66       1          0 0.0200 0.367  0.0202 0.0200  0.572                 1                  0
    ##  7            1   1500       7.31       1          0 0.0222 0.0226 0.133  0.133   0.689                 1                  0
    ##  8            1   1800       7.50       1          0 0.0223 0.362  0.0222 0.0223  0.571                 1                  0
    ##  9            1   3900       8.27       1          0 0.0286 0.200  0.0286 0.0287  0.714                 1                  0
    ## 10            1   7700       8.95       1          0 0.0200 0.115  0.0200 0.0200  0.825                 1                  0
    ## # ... with 5,135 more rows, and 11 more variables: weekday_is_wednesday <dbl>, weekday_is_thursday <dbl>,
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

When a subset of data is selected for the tech channel articles which
contain 7346 articles, the subset of data is then split into a training
set (70% of the subset data) and a test set (30% of the subset data)
based on the target variable, the number of shares. There are 5145
articles in the training set and 2201 observations in the test set
regarding the tech channel. The `createDataPartition` function from the
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

The tech channel has 5145 articles collected. Now let us take a look at
the relationships between our response and the predictors with some
numerical summaries and plots.

## 3.1. Numerical Summaries

Table 2 shows the popularity of the news articles on different days of
the week. We classified the number of shares greater than 1400 in a day
as “popular” and the number of shares less than 1400 in a day as
“unpopular”. We can see the number of articles from the tech channel
classified into “popular” group or “unpopular” group on different days
of the week from January 7th, 2013 to January 7th, 2015 when the
articles were published and retrieved by the study. Note, this table may
not reflect on the information contained in the data due to
dichotomizing the data.

Table 3 shows the average shares of the articles on different days of
the week. We can compare and determine which day of the week has the
most average number of shares for the tech channel. Here, we can see a
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

Table 4 shows the numerical summaries of *average keywords* from tech
channel in mashable.com on different days of the week. This table
indicates the number of times *average keywords* shown in the articles
regarding the average number of shares, and the table is showing the
average number of those *average keywords* calculated for each day of
the week so that we can compare to see which day of the week, the
*average keywords* showed up the most or the worst according to the
average of shares in the tech channel.

Table 5 shows the numerical summaries of average shares of referenced
articles in mashable.com on different days of the week. We calculated
the average number of shares of those articles that contained the
earlier popularity of news referenced for each day of the week so that
we can compare which day has the most or the worst average number of
shares when there were earlier popularity of news referenced in the
techarticles.

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
| Unpopular |    326 |     413 |       430 |      362 |    221 |       59 |     51 |
| Popular   |    503 |     614 |       606 |      535 |    472 |      323 |    230 |

Table 2. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 3. Average Shares vs. Average Log(shares) on Day of the Week")
```

| day.week  | Avg.shares | Sd.shares | Median.shares | Avg.logshares | Sd.logshares | Median.logshares |
|:----------|-----------:|----------:|--------------:|--------------:|-------------:|-----------------:|
| Monday    |   2908.466 |  4253.489 |          1600 |        7.5448 |       0.8311 |           7.3778 |
| Tuesday   |   2899.634 |  5162.654 |          1600 |        7.5184 |       0.8208 |           7.3778 |
| Wednesday |   3562.874 | 21010.806 |          1600 |        7.5367 |       0.8637 |           7.3778 |
| Thursday  |   2747.845 |  4294.657 |          1600 |        7.4975 |       0.7968 |           7.3778 |
| Friday    |   3164.838 |  5920.596 |          1800 |        7.6145 |       0.8060 |           7.4955 |
| Saturday  |   3750.105 |  6010.826 |          2300 |        7.8771 |       0.7426 |           7.7407 |
| Sunday    |   3692.751 |  4119.956 |          2300 |        7.8667 |       0.7663 |           7.7407 |

Table 3. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average Keywords on Day of the Week")
```

| day.week  | Avg.keyword | Sd.keyword | Median.keyword | IQR.keyword |
|:----------|------------:|-----------:|---------------:|------------:|
| Monday    |    2681.119 |   812.6908 |       2659.370 |    804.1680 |
| Tuesday   |    2769.896 |   693.9189 |       2708.336 |    727.7346 |
| Wednesday |    2708.965 |   776.8135 |       2658.664 |    731.7624 |
| Thursday  |    2775.331 |   836.8488 |       2721.222 |    703.3169 |
| Friday    |    2760.874 |   615.6428 |       2701.512 |    720.6229 |
| Saturday  |    2868.073 |   633.2257 |       2817.810 |    746.2075 |
| Sunday    |    2791.215 |   564.6896 |       2767.627 |    691.5831 |

Table 4. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 5. Summary of Average shares of referenced articles in Mashable on Day of the Week")
```

| day.week  | Avg.reference | Sd.reference | Median.reference | IQR.reference |
|:----------|--------------:|-------------:|-----------------:|--------------:|
| Monday    |      5783.178 |    13788.070 |         2833.333 |      3750.000 |
| Tuesday   |      7051.978 |    37302.274 |         2800.000 |      3750.000 |
| Wednesday |      6922.696 |    29799.508 |         2671.372 |      3594.682 |
| Thursday  |      8171.512 |    38021.828 |         2816.667 |      4080.000 |
| Friday    |      9233.143 |    49155.884 |         2900.000 |      3972.750 |
| Saturday  |      6766.594 |    20936.535 |         2922.000 |      3453.756 |
| Sunday    |      4432.695 |     5580.107 |         2660.000 |      3892.667 |

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
| Unpopular    |           0.4536 |          0.0752 |              0.4560 |
| Popular      |           0.4579 |          0.0748 |              0.4606 |

Table 6. Comparing Global Subjectivity between Popular and Unpopular
Articles

``` r
edadata %>% group_by(day.week) %>% summarize(
  Avg.images = mean(num_imgs), Sd.images = sd(num_imgs), Median.images = median(num_imgs), Avg.log.images = mean(log(num_imgs + 1)), Sd.log.images = sd(log(num_imgs + 1)), Median.log.images = median(log(num_imgs + 1))) %>%
  kable(digits = 4, caption = "Table 7. Comparing Image Counts by the Day of the Week")
```

| day.week  | Avg.images | Sd.images | Median.images | Avg.log.images | Sd.log.images | Median.log.images |
|:----------|-----------:|----------:|--------------:|---------------:|--------------:|------------------:|
| Monday    |     5.4572 |    8.8006 |             1 |         1.2199 |        1.0447 |            0.6931 |
| Tuesday   |     4.3262 |    6.9044 |             1 |         1.1509 |        0.9281 |            0.6931 |
| Wednesday |     4.3465 |    6.7674 |             1 |         1.1528 |        0.9351 |            0.6931 |
| Thursday  |     4.1159 |    7.0783 |             1 |         1.0978 |        0.9153 |            0.6931 |
| Friday    |     3.9351 |    6.2886 |             1 |         1.0635 |        0.9396 |            0.6931 |
| Saturday  |     5.2356 |    6.8153 |             2 |         1.3446 |        0.9888 |            1.0986 |
| Sunday    |     4.0107 |    6.2607 |             1 |         1.0753 |        0.9737 |            0.6931 |

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/tech_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

### 3.2.2. Boxplot

Figure 2 shows the number of shares across different days of the week.
Here, due to the huge number of large-valued outliers, I capped the
number of shares to 8,000 so that we can see the medians and the
interquartile ranges clearly for different days of the week.

This is a boxplot with the days of the week on the x-axis and the number
of shares on the y-axis. We can inspect the trend of shares to see if
the shares are higher on a Monday, a Friday or a Sunday for the tech
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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/tech_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/tech_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

### 3.2.3.Barplot

Figure 4 shows the popularity of the news articles in relations to their
closeness to a top LDA topic for the tech channel on any day of the
week. The Latent Dirichlet Allocation (LDA) is an algorithm applied to
the Mashable texts of the articles in order to identify the five top
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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/tech_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

### 3.2.4. Line Plot

Figure 5 is a line plot that shows the same measurements as in Figure 4
that we can see the patterns of the mean ratios of a LDA topic vary or
not vary across time in different popularity groups more clearly. Again,
some mean ratios of LDA topics do not seem to vary across time when the
corresponding lines are flattened while other mean ratios of LDA topics
vary across time when their lines are fluctuating. The patterns observed
in the “popular” group may not reflect on the same trend in the
“unpopular” group for articles in the tech channel.

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/tech_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

### 3.2.5. Scatterplots

Figure 6 shows the relationship between the average keyword and
log-transformed shares for articles in the tech channel across different
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
average keywords in the articles from the tech channel.

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/tech_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/tech_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

### 3.2.6. QQ Plots

To justify the usage of the log transformations for shares and images,
we’ll show the QQ plot of each over the tech channel in figures 8a, 8b,
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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/tech_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/tech_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/tech_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/tech_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

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

    ## # A tibble: 5,145 x 10
    ##    log.shares dayweek kw_avg_avg LDA_02 self_reference_avg~ average_token_len~ n_tokens_content n_tokens_title global_subjecti~
    ##         <dbl> <fct>        <dbl>  <dbl>               <dbl>              <dbl>            <dbl>          <dbl>            <dbl>
    ##  1     -1.65  Monday       -3.72 -0.569             -0.120               0.289            0.999         1.33              0.762
    ##  2     -1.01  Monday       -3.72 -0.613              0.0408             -0.699           -0.411        -0.0917           -0.253
    ##  3      2.63  Monday       -3.72 -0.594             -0.130               0.392            1.27         -1.04              1.11 
    ##  4      0.432 Monday       -3.72 -0.628             -0.120               0.300            1.35          1.33              0.669
    ##  5     -1.80  Monday       -3.72 -0.569             -0.120               0.127            1.16          0.382             1.03 
    ##  6     -1.11  Monday       -3.72 -0.627             -0.188              -1.00            -0.620        -1.04             -1.91 
    ##  7     -0.325 Monday       -3.72  0.161             -0.140               0.593           -0.490        -1.04              1.39 
    ##  8     -0.104 Monday       -3.72 -0.613             -0.124               0.145            1.31          0.855             0.326
    ##  9      0.834 Monday       -3.72 -0.569             -0.216               1.21             0.117        -0.0917            1.01 
    ## 10      1.66  Monday       -3.72 -0.629             -0.104               0.501            1.34         -0.0917            0.738
    ## # ... with 5,135 more rows, and 1 more variable: num_imgs <dbl>

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
    ## -4.7569 -0.6355 -0.1733  0.4783  6.6963 
    ## 
    ## Coefficients:
    ##                                                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                    0.008098   0.034479   0.235  0.81433    
    ## dayweekTuesday                                -0.068369   0.045390  -1.506  0.13206    
    ## dayweekWednesday                              -0.044501   0.045318  -0.982  0.32615    
    ## dayweekThursday                               -0.097820   0.046845  -2.088  0.03683 *  
    ## dayweekFriday                                  0.054307   0.050049   1.085  0.27793    
    ## dayweekSaturday                                0.312553   0.060198   5.192 2.16e-07 ***
    ## dayweekSunday                                  0.415197   0.072755   5.707 1.22e-08 ***
    ## kw_avg_avg                                     0.149271   0.013642  10.942  < 2e-16 ***
    ## LDA_02                                        -0.014146   0.013927  -1.016  0.30979    
    ## self_reference_avg_sharess                     0.252592   0.080976   3.119  0.00182 ** 
    ## average_token_length                          -0.001203   0.018259  -0.066  0.94746    
    ## n_tokens_content                               0.157037   0.022982   6.833 9.28e-12 ***
    ## n_tokens_title                                -0.031315   0.013645  -2.295  0.02177 *  
    ## global_subjectivity                            0.014622   0.014138   1.034  0.30107    
    ## num_imgs                                      -0.003432   0.016432  -0.209  0.83456    
    ## `I(n_tokens_content^2)`                       -0.013582   0.005738  -2.367  0.01796 *  
    ## `kw_avg_avg:num_imgs`                          0.058856   0.014035   4.194 2.79e-05 ***
    ## `average_token_length:global_subjectivity`     0.010258   0.004556   2.252  0.02438 *  
    ## `dayweekTuesday:self_reference_avg_sharess`   -0.207332   0.085247  -2.432  0.01504 *  
    ## `dayweekWednesday:self_reference_avg_sharess` -0.163479   0.087570  -1.867  0.06198 .  
    ## `dayweekThursday:self_reference_avg_sharess`  -0.231091   0.085714  -2.696  0.00704 ** 
    ## `dayweekFriday:self_reference_avg_sharess`    -0.232065   0.084637  -2.742  0.00613 ** 
    ## `dayweekSaturday:self_reference_avg_sharess`  -0.216320   0.112682  -1.920  0.05495 .  
    ## `dayweekSunday:self_reference_avg_sharess`     0.830880   0.352614   2.356  0.01849 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9664 on 5121 degrees of freedom
    ## Multiple R-squared:  0.07022,    Adjusted R-squared:  0.06604 
    ## F-statistic: 16.82 on 23 and 5121 DF,  p-value: < 2.2e-16

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
    ## -4.7003 -0.6282 -0.1727  0.4831  6.6249 
    ## 
    ## Coefficients:
    ##                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                 0.024250   0.034460   0.704   0.4816    
    ## dayweekTuesday                             -0.054890   0.045030  -1.219   0.2229    
    ## dayweekWednesday                           -0.033134   0.044941  -0.737   0.4610    
    ## dayweekThursday                            -0.088357   0.046483  -1.901   0.0574 .  
    ## dayweekFriday                               0.063557   0.049674   1.279   0.2008    
    ## dayweekSaturday                             0.320178   0.059799   5.354 8.97e-08 ***
    ## dayweekSunday                               0.348873   0.066673   5.233 1.74e-07 ***
    ## kw_avg_avg                                  0.169933   0.014865  11.432  < 2e-16 ***
    ## LDA_02                                     -0.016344   0.013890  -1.177   0.2394    
    ## average_token_length                       -0.002165   0.018176  -0.119   0.9052    
    ## n_tokens_content                            0.159362   0.023807   6.694 2.41e-11 ***
    ## n_tokens_title                             -0.032123   0.013582  -2.365   0.0181 *  
    ## global_subjectivity                         0.014458   0.014058   1.028   0.3038    
    ## `I(log(num_imgs + 1))`                     -0.006973   0.024282  -0.287   0.7740    
    ## `I(n_tokens_content^2)`                    -0.013589   0.005830  -2.331   0.0198 *  
    ## `I(log(self_reference_avg_sharess + 1))`    0.406629   0.050624   8.032 1.18e-15 ***
    ## `kw_avg_avg:I(log(num_imgs + 1))`           0.095014   0.018832   5.045 4.68e-07 ***
    ## `average_token_length:global_subjectivity`  0.010861   0.004535   2.395   0.0167 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9622 on 5127 degrees of freedom
    ## Multiple R-squared:  0.07718,    Adjusted R-squared:  0.07412 
    ## F-statistic: 25.22 on 17 and 5127 DF,  p-value: < 2.2e-16

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

    ## # A tibble: 5,145 x 16
    ##    log.shares kw_avg_avg LDA_02 weekday_is_monday weekday_is_tuesday weekday_is_wednesday weekday_is_thursday weekday_is_friday
    ##         <dbl>      <dbl>  <dbl>             <dbl>              <dbl>                <dbl>               <dbl>             <dbl>
    ##  1     -1.65       -3.72 -0.569              2.28             -0.499               -0.502              -0.459            -0.394
    ##  2     -1.01       -3.72 -0.613              2.28             -0.499               -0.502              -0.459            -0.394
    ##  3      2.63       -3.72 -0.594              2.28             -0.499               -0.502              -0.459            -0.394
    ##  4      0.432      -3.72 -0.628              2.28             -0.499               -0.502              -0.459            -0.394
    ##  5     -1.80       -3.72 -0.569              2.28             -0.499               -0.502              -0.459            -0.394
    ##  6     -1.11       -3.72 -0.627              2.28             -0.499               -0.502              -0.459            -0.394
    ##  7     -0.325      -3.72  0.161              2.28             -0.499               -0.502              -0.459            -0.394
    ##  8     -0.104      -3.72 -0.613              2.28             -0.499               -0.502              -0.459            -0.394
    ##  9      0.834      -3.72 -0.569              2.28             -0.499               -0.502              -0.459            -0.394
    ## 10      1.66       -3.72 -0.629              2.28             -0.499               -0.502              -0.459            -0.394
    ## # ... with 5,135 more rows, and 8 more variables: weekday_is_saturday <dbl>, weekday_is_sunday <dbl>,
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
    ## 5145 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 4117, 4117, 4116, 4116, 4114 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE       Rsquared    MAE      
    ##   1     0.9681007  0.09337968  0.7376959
    ##   2     0.9526500  0.09454871  0.7244869
    ##   3     0.9545165  0.08982619  0.7269944
    ##   4     0.9562437  0.08794711  0.7280041
    ##   5     0.9584549  0.08513957  0.7299794
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 2.

``` r
random_forest$results
```

    ## # A tibble: 5 x 7
    ##    mtry  RMSE Rsquared   MAE RMSESD RsquaredSD   MAESD
    ##   <int> <dbl>    <dbl> <dbl>  <dbl>      <dbl>   <dbl>
    ## 1     1 0.968   0.0934 0.738 0.0184     0.0127 0.00893
    ## 2     2 0.953   0.0945 0.724 0.0158     0.0119 0.00676
    ## 3     3 0.955   0.0898 0.727 0.0163     0.0120 0.00715
    ## 4     4 0.956   0.0879 0.728 0.0160     0.0153 0.00719
    ## 5     5 0.958   0.0851 0.730 0.0166     0.0136 0.00696

``` r
# mtry = random_forest$bestTune[[1]]
```

We then used 5 fold cross validation to search for the tuning parameter
value ranging from 1 to 5 that produces the optimal random forest model.
The optimal model chosen by cross validation produced the smallest RMSE
value when mtry = 2 and the lowest RMSE = 0.95265 when training the
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
    ## 5145 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4631, 4629, 4630, 4632, 4631, 4630, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9680498  0.07457499  0.7391498
    ##   1                   50      0.9602869  0.08192167  0.7311762
    ##   1                   75      0.9577283  0.08448760  0.7287814
    ##   1                  100      0.9572436  0.08463300  0.7277708
    ##   1                  125      0.9568957  0.08528805  0.7275686
    ##   2                   25      0.9598923  0.08583849  0.7321364
    ##   2                   50      0.9553956  0.08927017  0.7274075
    ##   2                   75      0.9547911  0.08909959  0.7261715
    ##   2                  100      0.9538882  0.09056110  0.7250397
    ##   2                  125      0.9538688  0.09101621  0.7252245
    ##   3                   25      0.9576599  0.08661822  0.7295097
    ##   3                   50      0.9541545  0.09040929  0.7256713
    ##   3                   75      0.9535309  0.09152453  0.7246301
    ##   3                  100      0.9541793  0.09088086  0.7248127
    ##   3                  125      0.9550433  0.08979720  0.7245155
    ##   4                   25      0.9561974  0.08939117  0.7287073
    ##   4                   50      0.9532018  0.09223397  0.7253743
    ##   4                   75      0.9520815  0.09446419  0.7248733
    ##   4                  100      0.9522816  0.09488994  0.7241029
    ##   4                  125      0.9532154  0.09360932  0.7241736
    ##   5                   25      0.9542755  0.09255264  0.7274480
    ##   5                   50      0.9500795  0.09813167  0.7228510
    ##   5                   75      0.9512828  0.09647459  0.7229434
    ##   5                  100      0.9533127  0.09387670  0.7239481
    ##   5                  125      0.9558907  0.09098614  0.7262094
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at
    ##  a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50, interaction.depth = 5, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
#boosted_tree$results
# n.trees = boosted_tree$bestTune[[1]]
```

We then used 10 fold cross validation to search all combinations of the
tuning parameters values using the `expand.grid` function to choose an
optimal model with the desired tuning parameters values. The optimal
boosted tree model chosen by cross validation produced the smallest RMSE
value (0.9500795) when n.trees = 50, interaction.depth = 5, shrinkage =
0.1 and n.minobsinnode = 10 when training the model with the training
set.

## 4.4. Model Comparisons

The best model fit to predict the number of shares for the tech channel
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
| Linear Regression 1 | 1.0369 |   0.0146 | 0.7317 |
| Linear Regression 2 | 1.0369 |   0.0146 | 0.7317 |
| Random Forest       | 0.9087 |   0.0842 | 0.7068 |
| Boosted Tree        | 0.9164 |   0.0728 | 0.7099 |

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
sets, from the tech channel. A variable importance plot is produced
along with a table containing a ranking metrics of the variable
importance when fitting the final model with the entire data set of the
tech articles.

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
Forest** model for the tech articles. We fit the entire data set, both
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
    ## 7346 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 7346, 7346, 7346, 7346, 7346, 7346, ... 
    ## Resampling results:
    ## 
    ##   RMSE       Rsquared    MAE      
    ##   0.9407797  0.08834466  0.7205445
    ## 
    ## Tuning parameter 'mtry' was held constant at a value of 2
    ## 
    ## [[2]]
    ## rf variable importance
    ## 
    ##                            Overall
    ## kw_avg_avg                 100.000
    ## n_tokens_content            93.413
    ## weekday_is_saturday         83.548
    ## weekday_is_sunday           67.939
    ## num_imgs                    65.945
    ## self_reference_avg_sharess  63.444
    ## average_token_length        22.924
    ## weekday_is_thursday         21.602
    ## global_subjectivity         19.887
    ## LDA_02                      17.467
    ## weekday_is_wednesday        15.503
    ## n_tokens_title              11.515
    ## weekday_is_monday           10.547
    ## weekday_is_tuesday           7.897
    ## weekday_is_friday            0.000
    ## 
    ## [[3]]

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/tech_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

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
