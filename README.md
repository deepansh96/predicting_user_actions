# Predicting_user_actions

In this course project(adapted from a UDEMY course), i implemented artifical neural networks for predicting user actions
based on an ecommerce company's user data.

# Data_set

The data included 5 features,namely-

•Is_mobile (0/1) (--If the user was browsing the site on mobile or not)

•N_products_viewed (int >=0) (--Number of products viewed on that e-commerce site)

•Visit_duration (real>=0) (--Number of hours for which the user was on that site)

•Is_returnign (0/1) (--If the user is a returning customer or a new one)

•Time_of_day (0/1/2/3)

The data given was of user action and recorded features of 500 users.

Time of day(24hrs) was divided into 4 buckets --
                       
                       (0 = 12am-6am)
                       (1 = 6am-12pm)
                       (2 = 12pm-6pm)
                       (3 = 6pm-12am)

# Prediction

•User_action (0/1/2/3) 

User actions were divided into 4 buckets --

                       (0 = If user bounced from the site)
                       (1 = If user added products to the cart)
                       (2 = If checkout was begun by the user)
                       (3 = If checkout was finished by the user)


# Goal

The goal was to train an artifical neural network on this data set and make predictions about the user action for any
incoming data. This would help a said ecommerce company in improving their sales and rating and also catering for 
their customers. 

For example - If by the incoming data of a user, the ANN predicts 0, i.e. the user is going to bounce, the company
can show a pop-up to keep the user engaged and not leave the site.

# Note

There are a few python scripts in the project.

process.py - pre-processes the data for the neural network to learn from

ann_predict.py - predicts the user_actions based on randomised weight matrices (bad score)

ann_train.py - trains the ANN using linear regression and softmax

ann_predict_after_train.py - uses the weights learned in ann_train.py to predict user_actions (Good score)

ecommerce_data.csv - comma seperated values for the ecommerce dataset
