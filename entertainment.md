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

    ## # A tibble: 4,941 x 23
    ##    class_shares shares log.shares dayweek kw_avg_avg LDA_00 LDA_01 LDA_02 LDA_03 LDA_04 weekday_is_monday weekday_is_tuesday
    ##           <dbl>  <dbl>      <dbl>   <dbl>      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>             <dbl>              <dbl>
    ##  1            0    593       6.39       1         0  0.500  0.378  0.0400 0.0413 0.0401                 1                  0
    ##  2            0   1200       7.09       1         0  0.0286 0.419  0.495  0.0289 0.0286                 1                  0
    ##  3            1   2100       7.65       1         0  0.0334 0.0345 0.215  0.684  0.0333                 1                  0
    ##  4            0   1200       7.09       1         0  0.126  0.0203 0.0200 0.814  0.0200                 1                  0
    ##  5            0   1200       7.09       1         0  0.0240 0.665  0.0225 0.266  0.0223                 1                  0
    ##  6            0    631       6.45       1         0  0.456  0.482  0.0200 0.0213 0.0200                 1                  0
    ##  7            0   1300       7.17       2      1114. 0.0500 0.525  0.324  0.0510 0.0500                 0                  1
    ##  8            1   6400       8.76       3       849. 0.0400 0.840  0.0400 0.0404 0.0400                 0                  0
    ##  9            0   1100       7.00       3       935. 0.0400 0.0418 0.258  0.620  0.0400                 0                  0
    ## 10            1   1500       7.31       3       827. 0.0333 0.677  0.0333 0.223  0.0335                 0                  0
    ## # ... with 4,931 more rows, and 11 more variables: weekday_is_wednesday <dbl>, weekday_is_thursday <dbl>,
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

When a subset of data is selected for the entertainment channel articles
which contain 7057 articles, the subset of data is then split into a
training set (70% of the subset data) and a test set (30% of the subset
data) based on the target variable, the number of shares. There are 4941
articles in the training set and 2116 observations in the test set
regarding the entertainment channel. The `createDataPartition` function
from the `caret` package is used to split the data into training and
test sets. We set a seed so that the analyses we implemented are
reproducible.

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

The entertainment channel has 4941 articles collected. Now let us take a
look at the relationships between our response and the predictors with
some numerical summaries and plots.

## 3.1. Numerical Summaries

Table 2 shows the popularity of the news articles on different days of
the week. We classified the number of shares greater than 1400 in a day
as “popular” and the number of shares less than 1400 in a day as
“unpopular”. We can see the number of articles from the entertainment
channel classified into “popular” group or “unpopular” group on
different days of the week from January 7th, 2013 to January 7th, 2015
when the articles were published and retrieved by the study. Note, this
table may not reflect on the information contained in the data due to
dichotomizing the data.

Table 3 shows the average shares of the articles on different days of
the week. We can compare and determine which day of the week has the
most average number of shares for the entertainment channel. Here, we
can see a potential problem for our analysis later. Median shares are
all very different from the average shares on any day of the week.
Recall that median is a robust measure for center. It is robust to
outliers in the data. On the contrary, mean is also a measure of center
but it is not robust to outliers. Mean measure can be influenced by
potential outliers.

In addition, Table 3 also shows the standard deviation of shares is huge
for any day of the week. They are potentially larger than the average
shares. This tells us the variance of shares for any day is huge. We
know a common variance stabilizing transformation to deal with
increasing variance of the response variable, that is, the
log-transformation, which could help us on this matter. Therefore, Table
3 again shows after the log-transformation of shares, the mean values
are similar to their corresponding median values, and their standard
deviations are much smaller than before relatively speaking.

Table 4 shows the numerical summaries of *average keywords* from
entertainment channel in mashable.com on different days of the week.
This table indicates the number of times *average keywords* shown in the
articles regarding the average number of shares, and the table is
showing the average number of those *average keywords* calculated for
each day of the week so that we can compare to see which day of the
week, the *average keywords* showed up the most or the worst according
to the average of shares in the entertainment channel.

Table 5 shows the numerical summaries of average shares of referenced
articles in mashable.com on different days of the week. We calculated
the average number of shares of those articles that contained the
earlier popularity of news referenced for each day of the week so that
we can compare which day has the most or the worst average number of
shares when there were earlier popularity of news referenced in the
entertainmentarticles.

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
| Unpopular |    614 |     566 |       576 |      536 |    392 |       97 |    134 |
| Popular   |    355 |     345 |       325 |      323 |    269 |      172 |    237 |

Table 2. Popularity on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.shares = mean(shares), Sd.shares = sd(shares), Median.shares = median(shares), 
  Avg.logshares = mean(log.shares), Sd.logshares = sd(log.shares), Median.logshares = median(log.shares)) %>% 
  kable(digits = 4, caption = "Table 3. Average Shares vs. Average Log(shares) on Day of the Week")
```

| day.week  | Avg.shares | Sd.shares | Median.shares | Avg.logshares | Sd.logshares | Median.logshares |
|:----------|-----------:|----------:|--------------:|--------------:|-------------:|-----------------:|
| Monday    |   2889.597 |  7029.259 |          1100 |        7.2639 |       0.9370 |           7.0031 |
| Tuesday   |   2877.996 |  7099.666 |          1100 |        7.2498 |       0.9504 |           7.0031 |
| Wednesday |   2693.796 |  6957.247 |          1100 |        7.2173 |       0.9289 |           7.0031 |
| Thursday  |   2654.277 |  5501.523 |          1100 |        7.2530 |       0.9217 |           7.0031 |
| Friday    |   2763.859 |  6564.225 |          1200 |        7.3003 |       0.8834 |           7.0901 |
| Saturday  |   3196.398 |  4899.815 |          1600 |        7.5907 |       0.8478 |           7.3778 |
| Sunday    |   3928.189 |  6540.015 |          1700 |        7.7251 |       0.9212 |           7.4384 |

Table 3. Average Shares vs. Average Log(shares) on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.keyword = mean(kw_avg_avg), Sd.keyword = sd(kw_avg_avg), Median.keyword = median(kw_avg_avg), 
  IQR.keyword = IQR(kw_avg_avg)) %>% 
  kable(digits = 4, caption = "Table 4. Summary of Average Keywords on Day of the Week")
```

| day.week  | Avg.keyword | Sd.keyword | Median.keyword | IQR.keyword |
|:----------|------------:|-----------:|---------------:|------------:|
| Monday    |    3138.246 |  1000.8555 |       2986.230 |    964.4435 |
| Tuesday   |    3196.072 |  1047.9397 |       2948.443 |   1092.8238 |
| Wednesday |    3095.430 |   919.6049 |       2924.109 |    977.7699 |
| Thursday  |    3131.609 |   978.9344 |       2946.254 |    983.3106 |
| Friday    |    3149.342 |   888.7727 |       3025.972 |   1010.1086 |
| Saturday  |    3250.077 |  1072.2700 |       3021.647 |   1069.6917 |
| Sunday    |    3212.291 |  1050.1105 |       3028.401 |    906.0124 |

Table 4. Summary of Average Keywords on Day of the Week

``` r
edadata %>% group_by(day.week) %>% summarise(
  Avg.reference = mean(self_reference_avg_sharess), Sd.reference = sd(self_reference_avg_sharess), 
  Median.reference = median(self_reference_avg_sharess), IQR.reference = IQR(self_reference_avg_sharess)) %>% 
  kable(digits = 4, caption = "Table 5. Summary of Average shares of referenced articles in Mashable on Day of the Week")
```

| day.week  | Avg.reference | Sd.reference | Median.reference | IQR.reference |
|:----------|--------------:|-------------:|-----------------:|--------------:|
| Monday    |      5005.183 |    10446.460 |             2000 |      3800.000 |
| Tuesday   |      5252.047 |     9066.755 |             2100 |      4292.298 |
| Wednesday |      4972.436 |     9763.866 |             2000 |      3776.500 |
| Thursday  |      4859.891 |    10757.457 |             2000 |      3773.750 |
| Friday    |      4483.275 |     8536.953 |             1980 |      3733.333 |
| Saturday  |      5377.535 |    19079.956 |             1850 |      3015.667 |
| Sunday    |      5213.800 |    11704.089 |             2200 |      3868.667 |

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
| Unpopular    |           0.4498 |          0.1112 |              0.4619 |
| Popular      |           0.4549 |          0.1164 |              0.4673 |

Table 6. Comparing Global Subjectivity between Popular and Unpopular
Articles

``` r
edadata %>% group_by(day.week) %>% summarize(
  Avg.images = mean(num_imgs), Sd.images = sd(num_imgs), Median.images = median(num_imgs), Avg.log.images = mean(log(num_imgs + 1)), Sd.log.images = sd(log(num_imgs + 1)), Median.log.images = median(log(num_imgs + 1))) %>%
  kable(digits = 4, caption = "Table 7. Comparing Image Counts by the Day of the Week")
```

| day.week  | Avg.images | Sd.images | Median.images | Avg.log.images | Sd.log.images | Median.log.images |
|:----------|-----------:|----------:|--------------:|---------------:|--------------:|------------------:|
| Monday    |     5.9690 |   10.7461 |             1 |         1.2408 |        1.0790 |            0.6931 |
| Tuesday   |     6.5543 |   11.5622 |             1 |         1.2622 |        1.1429 |            0.6931 |
| Wednesday |     5.3041 |   10.0689 |             1 |         1.1316 |        1.0681 |            0.6931 |
| Thursday  |     6.0221 |   10.4489 |             1 |         1.2352 |        1.0990 |            0.6931 |
| Friday    |     6.0908 |   12.3168 |             1 |         1.1603 |        1.1109 |            0.6931 |
| Saturday  |     6.9591 |   12.6545 |             1 |         1.3676 |        1.1102 |            0.6931 |
| Sunday    |     8.1536 |   13.8876 |             1 |         1.4512 |        1.1846 |            0.6931 |

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/entertainment_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

### 3.2.2. Boxplot

Figure 2 shows the number of shares across different days of the week.
Here, due to the huge number of large-valued outliers, I capped the
number of shares to 8,000 so that we can see the medians and the
interquartile ranges clearly for different days of the week.

This is a boxplot with the days of the week on the x-axis and the number
of shares on the y-axis. We can inspect the trend of shares to see if
the shares are higher on a Monday, a Friday or a Sunday for the
entertainment articles.

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/entertainment_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/entertainment_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

### 3.2.3.Barplot

Figure 4 shows the popularity of the news articles in relations to their
closeness to a top LDA topic for the entertainment channel on any day of
the week. The Latent Dirichlet Allocation (LDA) is an algorithm applied
to the Mashable texts of the articles in order to identify the five top
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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/entertainment_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

### 3.2.4. Line Plot

Figure 5 is a line plot that shows the same measurements as in Figure 4
that we can see the patterns of the mean ratios of a LDA topic vary or
not vary across time in different popularity groups more clearly. Again,
some mean ratios of LDA topics do not seem to vary across time when the
corresponding lines are flattened while other mean ratios of LDA topics
vary across time when their lines are fluctuating. The patterns observed
in the “popular” group may not reflect on the same trend in the
“unpopular” group for articles in the entertainment channel.

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/entertainment_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

### 3.2.5. Scatterplots

Figure 6 shows the relationship between the average keyword and
log-transformed shares for articles in the entertainment channel across
different days of the week. In the news popularity study, it showed
average keyword was ranked top one predictor in variable importance in
the optimal predictive model (random forest) they selected that produced
the highest accuracy in prediction of popularity online articles.
Therefore, we are interested to see how average keyword is related with
log shares. The different colored linear regression lines indicate
different days of the week.

If the points display an upward trend, it indicates a positive
relationship between the average keyword and log-shares. With an
increasing log number of shares, the number of average keywords also
increases, meaning people tend to share the article more when they see
more of those average keywords in the article. On the contrary, if the
points are in a downward trend, it indicates a negative relationship
between the average keyword and log-shares. With an decreasing log
number of shares, the number of average keywords decreases as well.
People tend to share the articles less when they see less of these
average keywords in the articles from the entertainment channel.

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/entertainment_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/entertainment_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

### 3.2.6. QQ Plots

To justify the usage of the log transformations for shares and images,
we’ll show the QQ plot of each over the entertainment channel in figures
8a, 8b, 8c, and 8d. We’re aiming for something close to a straight line,
which would indicate that the data is approximately normal in its
distribution and does not need further standardization.

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/entertainment_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/entertainment_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/entertainment_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

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

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/entertainment_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

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

    ## # A tibble: 4,941 x 10
    ##    log.shares dayweek   kw_avg_avg LDA_02 self_reference_av~ average_token_le~ n_tokens_content n_tokens_title global_subjecti~
    ##         <dbl> <fct>          <dbl>  <dbl>              <dbl>             <dbl>            <dbl>          <dbl>            <dbl>
    ##  1   -0.989   Monday         -3.20 -0.375             -0.421           0.257             -0.728         0.465             0.615
    ##  2   -0.234   Monday         -3.20  3.07              -0.467          -0.0744            -0.139        -0.978            -0.194
    ##  3    0.366   Monday         -3.20  0.952              0.123           0.0649            -0.776         1.43             -0.497
    ##  4   -0.234   Monday         -3.20 -0.527              0.307          -0.0160            -0.838         0.465             1.06 
    ##  5   -0.234   Monday         -3.20 -0.508             -0.345           0.710             -0.808         0.465             1.08 
    ##  6   -0.923   Monday         -3.20 -0.527             -0.270           0.00623           -0.470        -2.90             -0.140
    ##  7   -0.148   Tuesday        -2.07  1.78              -0.378           0.171             -0.611        -0.0160           -0.156
    ##  8    1.56    Wednesday      -2.34 -0.375             -0.215           0.462             -0.687        -2.42              0.585
    ##  9   -0.327   Wednesday      -2.25  1.28              -0.467           0.404             -0.483        -0.0160           -0.110
    ## 10    0.00540 Wednesday      -2.36 -0.426             -0.295          -0.0352            -0.265         0.946            -0.228
    ## # ... with 4,931 more rows, and 1 more variable: num_imgs <dbl>

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
    ## -3.9592 -0.5991 -0.2379  0.3679  4.5103 
    ## 
    ## Coefficients:
    ##                                                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                   -0.062687   0.032092  -1.953 0.050837 .  
    ## dayweekTuesday                                -0.031163   0.044299  -0.703 0.481790    
    ## dayweekWednesday                              -0.040931   0.044439  -0.921 0.357061    
    ## dayweekThursday                               -0.012187   0.045000  -0.271 0.786536    
    ## dayweekFriday                                  0.040930   0.048463   0.845 0.398395    
    ## dayweekSaturday                                0.340453   0.066301   5.135 2.93e-07 ***
    ## dayweekSunday                                  0.475131   0.058704   8.094 7.24e-16 ***
    ## kw_avg_avg                                     0.178338   0.014200  12.559  < 2e-16 ***
    ## LDA_02                                        -0.033894   0.013940  -2.431 0.015076 *  
    ## self_reference_avg_sharess                     0.121476   0.031641   3.839 0.000125 ***
    ## average_token_length                           0.017702   0.043895   0.403 0.686763    
    ## n_tokens_content                              -0.028525   0.020922  -1.363 0.172821    
    ## n_tokens_title                                -0.002402   0.013774  -0.174 0.861588    
    ## global_subjectivity                            0.060294   0.018758   3.214 0.001316 ** 
    ## num_imgs                                       0.039491   0.015823   2.496 0.012601 *  
    ## `I(n_tokens_content^2)`                        0.006502   0.005444   1.194 0.232427    
    ## `kw_avg_avg:num_imgs`                          0.048956   0.016256   3.012 0.002612 ** 
    ## `average_token_length:global_subjectivity`     0.014620   0.012327   1.186 0.235703    
    ## `dayweekTuesday:self_reference_avg_sharess`    0.001241   0.048953   0.025 0.979774    
    ## `dayweekWednesday:self_reference_avg_sharess`  0.014674   0.047087   0.312 0.755333    
    ## `dayweekThursday:self_reference_avg_sharess`   0.027502   0.045289   0.607 0.543705    
    ## `dayweekFriday:self_reference_avg_sharess`    -0.044841   0.056327  -0.796 0.426019    
    ## `dayweekSaturday:self_reference_avg_sharess`  -0.136792   0.045555  -3.003 0.002689 ** 
    ## `dayweekSunday:self_reference_avg_sharess`    -0.113790   0.055358  -2.056 0.039880 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9593 on 4917 degrees of freedom
    ## Multiple R-squared:  0.08411,    Adjusted R-squared:  0.07982 
    ## F-statistic: 19.63 on 23 and 4917 DF,  p-value: < 2.2e-16

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
    ## -3.8611 -0.5944 -0.2360  0.3673  4.4929 
    ## 
    ## Coefficients:
    ##                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                -0.010562   0.033112  -0.319  0.74976    
    ## dayweekTuesday                             -0.035394   0.044232  -0.800  0.42363    
    ## dayweekWednesday                           -0.041963   0.044376  -0.946  0.34439    
    ## dayweekThursday                            -0.011871   0.044939  -0.264  0.79168    
    ## dayweekFriday                               0.044492   0.048356   0.920  0.35756    
    ## dayweekSaturday                             0.344647   0.066197   5.206 2.00e-07 ***
    ## dayweekSunday                               0.470791   0.058602   8.034 1.17e-15 ***
    ## kw_avg_avg                                  0.187084   0.015682  11.930  < 2e-16 ***
    ## LDA_02                                     -0.030357   0.013922  -2.181  0.02926 *  
    ## average_token_length                        0.015674   0.043758   0.358  0.72021    
    ## n_tokens_content                           -0.031643   0.021494  -1.472  0.14104    
    ## n_tokens_title                             -0.004075   0.013747  -0.296  0.76690    
    ## global_subjectivity                         0.055868   0.018663   2.993  0.00277 ** 
    ## `I(log(num_imgs + 1))`                      0.052151   0.025256   2.065  0.03899 *  
    ## `I(n_tokens_content^2)`                     0.008332   0.005478   1.521  0.12835    
    ## `I(log(self_reference_avg_sharess + 1))`    0.253738   0.030280   8.380  < 2e-16 ***
    ## `kw_avg_avg:I(log(num_imgs + 1))`           0.068476   0.023217   2.949  0.00320 ** 
    ## `average_token_length:global_subjectivity`  0.017276   0.012330   1.401  0.16124    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.9579 on 4923 degrees of freedom
    ## Multiple R-squared:  0.08561,    Adjusted R-squared:  0.08245 
    ## F-statistic: 27.11 on 17 and 4923 DF,  p-value: < 2.2e-16

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

    ## # A tibble: 4,941 x 16
    ##    log.shares kw_avg_avg LDA_02 weekday_is_monday weekday_is_tuesday weekday_is_wednesday weekday_is_thursday weekday_is_friday
    ##         <dbl>      <dbl>  <dbl>             <dbl>              <dbl>                <dbl>               <dbl>             <dbl>
    ##  1   -0.989        -3.20 -0.375             2.02              -0.475               -0.472              -0.459            -0.393
    ##  2   -0.234        -3.20  3.07              2.02              -0.475               -0.472              -0.459            -0.393
    ##  3    0.366        -3.20  0.952             2.02              -0.475               -0.472              -0.459            -0.393
    ##  4   -0.234        -3.20 -0.527             2.02              -0.475               -0.472              -0.459            -0.393
    ##  5   -0.234        -3.20 -0.508             2.02              -0.475               -0.472              -0.459            -0.393
    ##  6   -0.923        -3.20 -0.527             2.02              -0.475               -0.472              -0.459            -0.393
    ##  7   -0.148        -2.07  1.78             -0.494              2.10                -0.472              -0.459            -0.393
    ##  8    1.56         -2.34 -0.375            -0.494             -0.475                2.12               -0.459            -0.393
    ##  9   -0.327        -2.25  1.28             -0.494             -0.475                2.12               -0.459            -0.393
    ## 10    0.00540      -2.36 -0.426            -0.494             -0.475                2.12               -0.459            -0.393
    ## # ... with 4,931 more rows, and 8 more variables: weekday_is_saturday <dbl>, weekday_is_sunday <dbl>,
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
    ## 4941 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 3953, 3953, 3951, 3953, 3954 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE       Rsquared    MAE      
    ##   1     0.9681094  0.09130723  0.7141474
    ##   2     0.9549070  0.09052784  0.7044485
    ##   3     0.9545385  0.09106667  0.7064112
    ##   4     0.9569081  0.08859225  0.7085530
    ##   5     0.9589594  0.08610059  0.7113393
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 3.

``` r
random_forest$results
```

    ## # A tibble: 5 x 7
    ##    mtry  RMSE Rsquared   MAE RMSESD RsquaredSD  MAESD
    ##   <int> <dbl>    <dbl> <dbl>  <dbl>      <dbl>  <dbl>
    ## 1     1 0.968   0.0913 0.714 0.0220     0.0225 0.0138
    ## 2     2 0.955   0.0905 0.704 0.0220     0.0235 0.0151
    ## 3     3 0.955   0.0911 0.706 0.0207     0.0266 0.0137
    ## 4     4 0.957   0.0886 0.709 0.0219     0.0268 0.0148
    ## 5     5 0.959   0.0861 0.711 0.0203     0.0242 0.0125

``` r
# mtry = random_forest$bestTune[[1]]
```

We then used 5 fold cross validation to search for the tuning parameter
value ranging from 1 to 5 that produces the optimal random forest model.
The optimal model chosen by cross validation produced the smallest RMSE
value when mtry = 3 and the lowest RMSE = 0.9545385 when training the
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
    ## 4941 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4447, 4447, 4446, 4447, 4448, 4446, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   25      0.9660600  0.07731253  0.7137139
    ##   1                   50      0.9599540  0.08327626  0.7073945
    ##   1                   75      0.9581056  0.08537964  0.7052304
    ##   1                  100      0.9580254  0.08537079  0.7051353
    ##   1                  125      0.9571132  0.08704182  0.7050659
    ##   2                   25      0.9602395  0.08487025  0.7079025
    ##   2                   50      0.9569327  0.08773652  0.7046863
    ##   2                   75      0.9563340  0.08877398  0.7045449
    ##   2                  100      0.9572178  0.08673437  0.7045267
    ##   2                  125      0.9570565  0.08727950  0.7043570
    ##   3                   25      0.9605892  0.08218739  0.7073099
    ##   3                   50      0.9578258  0.08561987  0.7044426
    ##   3                   75      0.9573493  0.08729866  0.7041164
    ##   3                  100      0.9598035  0.08365975  0.7064332
    ##   3                  125      0.9610540  0.08159043  0.7064063
    ##   4                   25      0.9585285  0.08501493  0.7057991
    ##   4                   50      0.9580140  0.08447550  0.7042615
    ##   4                   75      0.9597635  0.08258576  0.7058715
    ##   4                  100      0.9605794  0.08191110  0.7070249
    ##   4                  125      0.9632163  0.07900014  0.7078124
    ##   5                   25      0.9603175  0.08134828  0.7074426
    ##   5                   50      0.9605449  0.08056887  0.7061059
    ##   5                   75      0.9620879  0.07929481  0.7065239
    ##   5                  100      0.9661335  0.07415559  0.7098351
    ##   5                  125      0.9678056  0.07300338  0.7114376
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at
    ##  a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 75, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
#boosted_tree$results
# n.trees = boosted_tree$bestTune[[1]]
```

We then used 10 fold cross validation to search all combinations of the
tuning parameters values using the `expand.grid` function to choose an
optimal model with the desired tuning parameters values. The optimal
boosted tree model chosen by cross validation produced the smallest RMSE
value (0.956334) when n.trees = 75, interaction.depth = 2, shrinkage =
0.1 and n.minobsinnode = 10 when training the model with the training
set.

## 4.4. Model Comparisons

The best model fit to predict the number of shares for the entertainment
channel can be determined by looking at the Root Mean Squared Error, the
Mean Absolute Error, or the R-squared value using the test set. Table 8
shows these criteria measures for each candidate model. The approach
I’ve taken below picks whichever model is considered superior by the
majority score of these three metrics, and if each of the metrics picks
a different model or the majority score is a tiebreaker, then the one
with the lowest RMSE will be chosen.

``` r
result2 <- rbind(cv_rmse1, cv_rmse1, rf_rmse, boost_rmse)
row.names(result2) <- c("Linear Regression 1", "Linear Regression 2", "Random Forest", "Boosted Tree")
kable(result2, digits = 4, caption = "Table 8. Cross Validation - Comparisons of the models in test set")
```

|                     |   RMSE | Rsquared |    MAE |
|:--------------------|-------:|---------:|-------:|
| Linear Regression 1 | 0.9760 |   0.0581 | 0.7014 |
| Linear Regression 2 | 0.9760 |   0.0581 | 0.7014 |
| Random Forest       | 0.9585 |   0.0820 | 0.6994 |
| Boosted Tree        | 0.9597 |   0.0786 | 0.6931 |

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
sets, from the entertainment channel. A variable importance plot is
produced along with a table containing a ranking metrics of the variable
importance when fitting the final model with the entire data set of the
entertainment articles.

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
Forest** model for the entertainment articles. We fit the entire data
set, both the training and the test set from the channel in the final
chosen model. The values of RMSE, MAE and R-squared are calculated with
the entire data set using the final model. A variable importance plot of
the top 10 most important variables and a table containing a ranking
metric of the relative variable importance are produced below. We can
examine which predictors contributed the most in predicting the
popularity of online news in the final model accordingly. Table 1
containing attribute information from the *Introduction* section is
copied below for comparisons of variable importance.

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
    ## 7057 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 7057, 7057, 7057, 7057, 7057, 7057, ... 
    ## Resampling results:
    ## 
    ##   RMSE       Rsquared    MAE     
    ##   0.9572481  0.08005009  0.705909
    ## 
    ## Tuning parameter 'mtry' was held constant at a value of 3
    ## 
    ## [[2]]
    ## rf variable importance
    ## 
    ##                            Overall
    ## weekday_is_sunday          100.000
    ## n_tokens_content            87.902
    ## kw_avg_avg                  84.506
    ## self_reference_avg_sharess  69.976
    ## average_token_length        55.376
    ## weekday_is_saturday         54.716
    ## num_imgs                    53.759
    ## LDA_02                      38.960
    ## global_subjectivity         38.387
    ## n_tokens_title              13.080
    ## weekday_is_wednesday        11.443
    ## weekday_is_tuesday           9.278
    ## weekday_is_friday            7.219
    ## weekday_is_thursday          3.073
    ## weekday_is_monday            0.000
    ## 
    ## [[3]]

![](C:/Users/peach/documents/ST558/ST558_repos/News-Popularity-Prediction/entertainment_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

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
