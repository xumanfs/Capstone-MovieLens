### Capstone MovieLens
### author: XuMan
### website: https://github.com/xumanfs/Capstone-MovieLens.git

################################
# Create edx set, validation set
################################

# Load packages

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(irlba)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

ratings <- fread(text = gsub("::", "\t", readLines(file("/Users/xuman/ds_projects/Capstone-MovieLens/ml-10M100K/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(file("/Users/xuman/ds_projects/Capstone-MovieLens/ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId], title = as.character(title), genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
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

rm(ratings, movies, test_index, temp, movielens, removed)

################################
# Data exploration and visualization
################################

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

# Distribution of average rating by release year
edx %>% mutate(year = str_sub(title, -5, -2)) %>%
  group_by(year) %>%
  summarize(ry_avgs = mean(rating)) %>%
  ggplot(aes(ry_avgs)) +
  geom_histogram(color = "white") +
  ggtitle("Distribution of average rating by release year") +
  xlab("Average rating") +
  ylab("Number of years")


# Distribution of average rating by genres
edx %>% group_by(genres) %>% 
  summarize(genres_avgs = mean(rating)) %>%
  ggplot(aes(genres_avgs)) +
  geom_histogram(color = "white") +
  ggtitle("Distribution of average rating by genres") +
  xlab("Average rating") +
  ylab("Number of genres")


################################
# Create and test models
################################

# Define RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Create train and test set
set.seed(2029)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Regularized movie and user effect model
lambdas_iu <- seq(0, 10, 0.25)
rmse_iu_cd <- sapply(lambdas_iu, function(lambda){
  mu <- mean(train_set$rating)
  
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
plot(lambdas_iu, rmse_iu_cd)

lambda_iu <- lambdas_iu[which.min(rmse_iu_cd)]
rmse_iu_test <- rmse_iu_cd[which.min(rmse_iu_cd)]

mu <- mean(train_set$rating)
b_i <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda_iu)) 
b_u <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda_iu))
predicted_ratings_iu_1 <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

max(predicted_ratings_iu_1)
min(predicted_ratings_iu_1)

## Adjust the movie and user effect model by confining the rating within 0.5 and 5
predicted_ratings_iu <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  mutate(pred_l = ifelse(pred>=0.5 & pred<=5, pred, ifelse(pred<0.5, 0.5, 5)))%>%
  pull(pred_l)

rmse_iu <- RMSE(predicted_ratings_iu, test_set$rating)
rmse_iu

train_set_iu <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by = 'userId')

# Model with movie, user and release_year effect
b_tr <- train_set_iu %>%
  mutate(year = str_sub(title, -5, -2)) %>%
  group_by(year) %>%
  summarize(b_tr = mean(rating - mu - b_i - b_u))

# Model with movie, user, release_year and genre effect
b_g <- train_set_iu %>%
  mutate(year = str_sub(title, -5, -2)) %>%
  left_join(b_tr, by = 'year') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_tr))

predicted_ratings_iugy <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(year = str_sub(title, -5, -2)) %>%
  left_join(b_tr, by = 'year') %>%
  left_join(b_g, by = 'genres') %>%
  mutate(pred = mu + b_i + b_u + b_tr + b_g) %>%
  mutate(pred_l = ifelse(pred>=0.5 & pred<=5, pred, ifelse(pred<0.5, 0.5, 5)))%>%
  pull(pred_l)

rmse <- RMSE(predicted_ratings_iugy, test_set$rating)

train_set_iugy <- train_set_iu %>%
  mutate(year = str_sub(title, -5, -2)) %>%
  left_join(b_tr, by = 'year') %>%
  left_join(b_g, by = 'genres')

################################
# Fianl test with validation set
################################

predicted_ratings_on_validationset_uigy <- validation %>%
  mutate(year = str_sub(title, -5, -2)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_tr, by = 'year') %>%
  left_join(b_g, by = 'genres') %>%
  mutate(b_i = ifelse(is.na(b_i), 0, b_i)) %>%
  mutate(b_u = ifelse(is.na(b_u), 0, b_u)) %>%
  mutate(b_tr = ifelse(is.na(b_tr), 0, b_tr))%>%
  mutate(b_g = ifelse(is.na(b_g), 0, b_g)) %>%
  mutate(pred = mu + b_i + b_u + b_tr + b_g) %>%
  mutate(pred_l = ifelse(pred>=0.5 & pred<=5, pred, ifelse(pred<0.5, 0.5, 5))) %>%
  pull(pred_l)

rmse_v <- RMSE(predicted_ratings_on_validationset_uigy, validation$rating)

print(rmse)