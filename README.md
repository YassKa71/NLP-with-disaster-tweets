# NLP-with-disaster-tweets
This project aims to find a solution to the "" Kaggle Competition. It's a good initiation to NLP techniques.

## Problem description
Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).
But, it’s not always clear whether a person’s words are actually announcing a disaster.

In this competition, we’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. We have access to a dataset of 10,000 tweets that were hand classified.


## Rigid Classifier

We start by a simple approach to solve this problem: We assume first that words present in each tweet are a great indicator of disasters. Therefore, we use first a **Count Vectorizer** to count the number of times each token appears in each tweet, and then we use a linear model (scikit-learn Rigid Classifier) to classify whether there is a real disaster or not.

Using **cross validation**, we obtain **0,55** as a median **F1 score**.

## Fine-tuning Bert

The first approach didn't take into account the context in the text within each tweet. Therefore, in this part we used the bidirectional property of bert models to catch the contexts within each tweet. We **fine-tuned a pretrained Bert model** after preprocessing the data.

This approach was better than the first one and the **F1 score** increased by 0,20 to reach **0,75**.

After analyzing some examples of inaccurate predictions, we can see that many improvements can be made on the preprocessing of the dataset.
