### Capstone MovieLens
### author: XuMan
### website: https://github.com/xumanfs/Capstone-MovieLens.git

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



################################
# Data exploration and visualization
################################

# Define RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Rating distribution
edx %>% ggplot(aes(rating)) +
  geom_histogram() +
  ggtitle("Rating Distribution") +
  xlab("Rating")+
  scale_x_continuous(breaks = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000)))

# Distribution of average rating by movies
edx %>% group_by(movieId) %>% 
  summarize(movie_avgs = mean(rating)) %>%
  ggplot(aes(movie_avgs)) +
  geom_histogram(color = "white") +
  ggtitle("Distribution of average rating by movies") +
  xlab("Average rating") +
  ylab("Number of movies")

# Distribution of average rating by users
edx %>% group_by(userId) %>% 
  summarize(user_avgs = mean(rating)) %>%
  ggplot(aes(user_avgs)) +
  geom_histogram(color = "white") +
  ggtitle("Distribution of average rating by users") +
  xlab("Average rating") +
  ylab("Number of users")

# Distribution of number of ratings per movie
edx %>% group_by(movieId) %>%
  summarize(number_of_rating_per_movie = n()) %>%
  ggplot(aes(number_of_rating_per_movie)) +
  geom_histogram(color = "white") +
  ggtitle("Distribution of number of ratings per movie") +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  scale_x_log10()

edx %>% group_by(movieId) %>%
  summarize(number_of_rating_per_movie = n()) %>%
  filter(number_of_rating_per_movie == 1) %>% nrow()

# Distribution of number of ratings per user
edx %>% group_by(userId) %>%
  summarize(number_of_rating_per_user = n()) %>%
  ggplot(aes(number_of_rating_per_user)) +
  geom_histogram(color = "white") +
  ggtitle("Distribution of number of ratings per user") +
  xlab("Number of ratings") +
  ylab("Number of users") +
  scale_x_log10()

edx %>% group_by(userId) %>%
  summarize(number_of_rating_per_user = n()) %>%
  filter(number_of_rating_per_user < 20) %>%
  nrow()

edx %>% group_by(userId) %>%
  summarize(number_of_rating_per_user = n()) %>%
  filter(number_of_rating_per_user > 2000) %>%
  nrow()

################################
# Create and test models
################################

# Create train and test set
set.seed(2029)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Regularized movie and user effect model
lambdas_iu_1 <- seq(0, 10, 1)
rmse_iu_1 <- sapply(lambdas_iu_1, function(lambda){
  b_i <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+lambda)) 
  
  b_u <- train_set %>% 
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+lambda))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  RMSE(predicted_ratings, test_set$rating)
})
plot(lambdas_iu_1, rmse_iu_1)

lambdas_iu_2 <- seq(4, 6, 0.1)
rmse_iu_2 <- sapply(lambdas_iu_2, function(lambda){
  b_i <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+lambda)) 
  
  b_u <- train_set %>% 
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+lambda))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  RMSE(predicted_ratings, test_set$rating)
})
plot(lambdas_iu_2, rmse_iu_2)

lambda_iu <- lambdas_iu_2[which.min(rmse_iu_2)]
rmse_iu <- rmse_iu_2[which.min(rmse_iu_2)]

################################
# Fianl test with validation set
################################

b_i <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda_iu)) 

b_u <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda_iu))

predicted_ratings_on_validationset <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

RMSE(predicted_ratings_on_validationset, validation$rating)