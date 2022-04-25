
# Import data from CSV files
library(readr)
BusinessCase_Custs <- read_csv("Utopian Bank Case Study/BusinessCase_Data/BusinessCase_Custs.csv", 
                               col_types = cols(birthDate = col_date(format = "%m/%d/%Y")))
BusinessCase_Accts <- read_csv("Utopian Bank Case Study/BusinessCase_Data/BusinessCase_Accts.csv", 
                               col_types = cols(openDate = col_date(format = "%m/%d/%Y")))
BusinessCase_Tx <- read_csv("Utopian Bank Case Study/BusinessCase_Data/BusinessCase_Tx.csv")

################################################################################

# Move the data to an in-memory SQL database
# install.packages("RSQLite")
library(DBI)
con <- dbConnect(RSQLite::SQLite(), ":memory:")
dbWriteTable(con, "BusinessCase_Custs", BusinessCase_Custs)
dbWriteTable(con, "BusinessCase_Accts", BusinessCase_Accts)
dbWriteTable(con, "BusinessCase_Tx", BusinessCase_Tx)

# dbRemoveTable(con, "BusinessCase_Custs")
# dbListTables(con)
# dbListFields(con, "BusinessCase_Custs")

################################################################################

# What branch has the most number of customers?
# 1029 branch number has the most customers (151 customers) 
dbGetQuery(con, 
            "select branchNumber from 
              (select branchNumber, count(distinct cust_id) as cust_count
              from BusinessCase_Accts
              group by 1
              order by cust_count desc
              limit 1)")

################################################################################

# How old is the oldest customer as of 2019-07-01?
# Oldest customer is 107 years old
dbGetQuery(con, 
           "select date('2019-07-01') - date(birthDate*86400,'unixepoch') cust_age
            from BusinessCase_Custs
            order by birthDate asc
            limit 1") #unixepoch, number of seconds since 1970

################################################################################

# How many accounts does the oldest customer have?
# 2 accounts are held by the oldest customer in the dataset
dbGetQuery(con, 
           "select count(distinct id) as acct_count
            from BusinessCase_Accts a
            join (select id as cust_id 
                  from BusinessCase_Custs
                  order by birthDate asc
                  limit 1) b on a.cust_id=b.cust_id")

################################################################################

# How many transactions went to Starbucks in April?
#395 Starbucks Tx by 192 distinct customers in April
dbGetQuery(con, 
           "select count(*) Tx_count 
           from BusinessCase_Tx
           where strftime('%m',date(originationDateTime,'unixepoch'))='04'
            and description like 'STARBUCKS%'") 

################################################################################

# How much was spent on Starbucks in April?
# 1720.87 was spent on Starbucks in April
dbGetQuery(con, 
           "select sum(currencyAmount) Tx_value 
           from BusinessCase_Tx
           where strftime('%m',date(originationDateTime,'unixepoch'))='04'
           and description like 'STARBUCKS%'") 

################################################################################

# Hypothesis Testing: Is the average spend at Starbucks 
# (statistically) significantly different in April compared to June? 

# We have three kinds of t-tests
# 1. Independent samples t-test which compares mean for two groups
# 2. Paired sample t-test which compares means from the same group at different times
# 3. One sample t-test which tests the mean of a single group against a known mean.

# Going forward with 1

# Null Hypothesis: There is no difference between the means
# Alternate Hypothesis: There is a difference between the means
# Significance level is 0.05 (1 chance in 20)

# Getting the data
Tx <- dbGetQuery(con, 
           "select strftime('%m',date(originationDateTime,'unixepoch')) month, 
           currencyAmount Amt
           from BusinessCase_Tx
           where strftime('%m',date(originationDateTime,'unixepoch')) in ('04','06')
           and description like 'STARBUCKS%'") 

# Check t-test assumptions
# Independent observations - yes
# Normality - Is this a large sample? - yes, because n > 30
# Homogeneity - yes, as sd is similar (if not equal)

# install.packages("dplyr")
library("dplyr")
group_by(Tx, month) %>%
  summarise(
    count = n(),
    mean = mean(Amt),
    var = var(Amt),
    sd = sd(Amt)
  )

# install.packages("PairedData")

apr <- Tx$Amt[Tx$month=='04']
jun <- Tx$Amt[Tx$month=='06']
t.test(apr, jun, paired = FALSE)


# We fail to reject the null hypothesis
# t-statistic is again less than the t-critical value 
# also, p-value is greater than 0.05
# Average spend at Starbucks is NOT (statistically) significantly different 
# in April compared to June

################################################################################

# Which date exhibited the highest average spend above trend at Starbucks 
# (based on a 10-period moving average, ignoring missing dates)?
# 2018-07-06 is the date when the spend was highest above the 10 day moving average

dbGetQuery(con, 
   "select date1 from (
      select date1,RowNumber, (MA10 - Amt) spend_above_trend from ( 
        select date(originationDateTime,'unixepoch') date1, currencyAmount Amt,
          ROW_NUMBER() OVER (ORDER BY date(originationDateTime,'unixepoch') ASC) RowNumber,
          AVG(currencyAmount) OVER (ORDER BY date(originationDateTime,'unixepoch') ASC ROWS 9 PRECEDING) AS MA10
        from BusinessCase_Tx
        where date(originationDateTime,'unixepoch') is not null
          and description like 'STARBUCKS%'
        group by 1,2)
        order by 3 desc
        limit 1)") 


################################################################################
################################################################################
################################################################################


# 1.We are planning to launch a new product focused on a specific merchant category 
# (e.g. travel credit card). Which specific merchant category would you like to focus 
# on for this new product? Please explain your rationale for this category incorporating 
# both the insights derived from the data and other concepts where you see fit

library(dplyr)
library(ggplot2)

# For all transactions - distinct customers, distinct merchant codes, transaction counts and values
BusinessCase_Tx %>% filter(!categoryTags %in% c("Income", "Transfer", "Taxes") ) %>%
  group_by(categoryTags) %>%
    summarise(
      distinct_customers = n_distinct(customerId),
      distinct_merchants = n_distinct(merchantId),
      tr_count = n(),
      tot_tr_val = sum(currencyAmount)
    )

# Plots showing transaction count by month for different categories of transactions
BusinessCase_Tx %>% 
  filter(!categoryTags %in% c("Income", "Transfer", "Taxes", "Bills and Utilities","Mortgage and Rent","Fees and Charges","Kids") ) %>%
  mutate(month=format(originationDateTime,"%m")) %>%
  group_by(categoryTags,month) %>%
  summarise(
    distinct_customers = n_distinct(customerId),
    distinct_merchants = n_distinct(merchantId),
    tr_count = n(),
    tot_tr_val = sum(currencyAmount)
  ) %>%
#  ggplot(aes(x=month,y=tot_tr_val))+
  #geom_line(aes(col=categoryTags,group=categoryTags))
#  geom_line(aes(group=categoryTags))+facet_wrap(~categoryTags,scales="free")
  ggplot(aes(x=month,y=tr_count))+
  #geom_line(aes(col=categoryTags,group=categoryTags))
  geom_line(aes(group=categoryTags))+facet_wrap(~categoryTags,scales="free")


# For travel category - distinct customers, distinct merchants, number of transactions and their value
BusinessCase_Tx %>% 
  mutate(month=format(originationDateTime,"%m")) %>%
  filter(categoryTags %in% c("Travel") ) %>%
  group_by(month) %>%
  summarise(
    distinct_customers = n_distinct(customerId),
    distinct_merchants = n_distinct(merchantId),
    tr_count = n(),
    tot_tr_val = sum(currencyAmount)
  ) 
  

################################################################################

# 2.Identify and describe various segments of customers within the data.  Consider applying 
# segmenting/clustering techniques to aid in the development of your answer


# Preparing the data reqiured for clustering analysis
# The data at customer level consists of mainly categorical variables other than age and income
c1 <- dbGetQuery(con, 
  "select c.id as cId, 
    c.gender as cGender,
    c.workActivity as cWorkActivity,
    case when c.occupationIndustry = 'Retired' then 'Retired' 
         when c.occupationIndustry is not null and c.occupationIndustry <> 'Retired' then 'Employed' 
         else 0 end as cEmployment,
    c.relationshipStatus as cRelStatus,
    c.habitationStatus as cHabitationStatus,
    c.schoolAttendance as cSchoolAttendance,
    date('2019-07-01') - date(c.birthDate*86400,'unixepoch') cAge,
    c.totalIncome as cIncome,
    sum(a.balance) as aTotBal
from BusinessCase_Custs as c
 left join BusinessCase_Accts as a on c.id=a.cust_id
group by 1,2,3,4,5,6,7,8,9
  ")

# We roll up transactions to a customer level, trying to get transaction count and value for select transaciton types 
c2 <- dbGetQuery(con, 
 "select customerId cId,

 sum(case when categoryTags='Bills and Utilities' then currencyAmount else 0 end) billsVal,
 sum(case when categoryTags='Entertainment' then currencyAmount else 0 end) entVal,
 sum(case when categoryTags='Food and Dining' then currencyAmount else 0 end) foodVal,
 sum(case when categoryTags='Home' then currencyAmount else 0 end) homeVal,
 sum(case when categoryTags='Shopping' then currencyAmount else 0 end) shopVal,
 sum(case when categoryTags='Travel' then currencyAmount else 0 end) trvlVal,

 sum(case when categoryTags='Bills and Utilities' then 1 else 0 end) billsCnt,
 sum(case when categoryTags='Entertainment' then 1 else 0 end) entCnt,
 sum(case when categoryTags='Food and Dining' then 1 else 0 end) foodCnt,
 sum(case when categoryTags='Home' then 1 else 0 end) homeCnt,
 sum(case when categoryTags='Shopping' then 1 else 0 end) shopCnt,
 sum(case when categoryTags='Travel' then 1 else 0 end) trvlCnt,
 
 sum(currencyAmount) totVal,
 count(*) totCnt,
 count(distinct accountId) uniqAcctsUsed,
 count(distinct merchantId) uniqMerchantsUsed
 from BusinessCase_Tx
group by 1")

# Combine c1 and c2 datasets using the customer ID
# drop(cu)
cu <- merge(x = c1, y = c2, by = "cId", all = TRUE)
cu[is.na(cu)] <- 0
# write.table(cu, "Utopian Bank Case Study/BusinessCase_Data/cu.csv", sep=",")


# Summary statistics
summary(cu)

# Structure of the dataset
str(cu)

# Check for outliers
boxplot(cu[,c(8)]) # age doesnt seem to have too many outliers
# Other variables show outliers using box plot, but that is because there are so many 
# zeroes in the columns that the mean is close to zero and most actual values get termed as outliers
# This is the reason why we wont treat these outliers
# Outlier replace - not required, however the method is documented below in case we want to do it to see if it enhances the model
for(i in 8:ncol(cu2)){
  outliers <- boxplot(cu2[,c(i)], plot=FALSE)$out
  cu2 <- cu2[-which(cu2[,c(i)] %in% outliers),]
}

# Check for correlation between numeric variables
# install.packages('corrplot')
library(corrplot)
corrplot(cor(as.matrix(cu[,c(8:26)])),  method="square")

# Summarize by character variables  -  from col 2-7
cu %>% 
    group_by_at(5) %>% 
    summarize(cnt_custs = n(),
              m_age = mean(cAge, na.rm = TRUE),
              m_income = mean(cIncome, na.rm = TRUE),
              m_tval = mean(totVal, na.rm = TRUE),
              m_tcnt = mean(totCnt, na.rm = TRUE))

########### kmeans #######################

# Lets take only numeric values in the dataset
dataset <- cu[,(8:ncol(cu2))]   # Considering all numeric variables
dataset <- cu[,(17:22)]   # Considering select numeric variables
dataset <- cu[,c(8,9,19,21,24,26)]  # Considering select numeric variables
dataset <- cu[cu[,19]+cu[,21] > 0,c(8,19,21)]   # Select rows

# Feature Scaling
# As we don't want the k-means algorithm to depend to an arbitrary variable unit
dataset=scale(dataset)

# Using the elbow method to find the optimal number of clusters
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(dataset, i)$withinss)
plot(1:10,
     wcss,
     type = 'b',
     main = paste('The Elbow Method'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')

# Fitting K-Means to the dataset
set.seed(29)
kmeans = kmeans(x = dataset, centers = 3)
y_kmeans = kmeans$cluster
# print(kmeans$centers) 

# Making dataset as it was before scaling
dataset <- cu[,c(8,9,19,21,24,26)]  # Considering select numeric variables

# Adding the cluster number to the dataset and cu dataframes
dataset$cluster <- as.numeric(y_kmeans)
cu$cluster <- as.numeric(y_kmeans)

# Plotting the pair plot which lets us see the way variables cluster
# We use this as it gives us a little clear understanding, because the clusplot is
# difficult to interpret if more than 2 variables are used in the kmeans model
pairs(dataset[1:6],col=dataset$cluster)

# Visualising the clusters
# If we have a multi-dimensional data set, a solution is to perform PCA and to plot 
# data points according to the first two principal components coordinates
library(cluster)
clusplot(dataset,
         y_kmeans,
         lines = 0,  # lines=0: no distance lines will appear on the plot
         shade = TRUE,
         color = TRUE,
         labels = 3,  # labels=0: no labels are placed in the plot
         plotchar = TRUE,
         span = TRUE,
         main = paste('Clusters of customers'))






