#!/usr/bin/env python
# coding: utf-8

# # **TikTok Project**
# **Course 2 - Get Started with Python**

# Welcome to the TikTok Project!
# 
# You have just started as a data professional at TikTok.
# 
# The team is still in the early stages of the project. You have received notice that TikTok's leadership team has approved the project proposal. To gain clear insights to prepare for a claims classification model, TikTok's provided data must be examined to begin the process of exploratory data analysis (EDA).
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # **Course 2 End-of-course project: Inspect and analyze data**
# 
# In this activity, you will examine data provided and prepare it for analysis.
# <br/>
# 
# **The purpose** of this project is to investigate and understand the data provided. This activity will:
# 
# 1.   Acquaint you with the data
# 
# 2.   Compile summary information about the data
# 
# 3.   Begin the process of EDA and reveal insights contained in the data
# 
# 4.   Prepare you for more in-depth EDA, hypothesis testing, and statistical analysis
# 
# **The goal** is to construct a dataframe in Python, perform a cursory inspection of the provided dataset, and inform TikTok data team members of your findings.
# <br/>
# *This activity has three parts:*
# 
# **Part 1:** Understand the situation
# * How can you best prepare to understand and organize the provided TikTok information?
# 
# **Part 2:** Understand the data
# 
# * Create a pandas dataframe for data learning and future exploratory data analysis (EDA) and statistical activities
# 
# * Compile summary information about the data to inform next steps
# 
# **Part 3:** Understand the variables
# 
# * Use insights from your examination of the summary data to guide deeper investigation into variables
# 
# <br/>
# 
# To complete the activity, follow the instructions and answer the questions below. Then, you will us your responses to these questions and the questions included in the Course 2 PACE Strategy Document to create an executive summary.
# 
# Be sure to complete this activity before moving on to Course 3. You can assess your work by comparing the results to a completed exemplar after completing the end-of-course project.

# # **Identify data types and compile summary information**
# 

# ### **Task 1. Understand the situation**
# 
# *   How can you best prepare to understand and organize the provided information?
# 
# 
# *Begin by exploring your dataset and consider reviewing the Data Dictionary.*

# ==> ENTER YOUR RESPONSE HERE
# Understanding the context of the case study which relates to claim raw data for a machine learning model.

# ### **Task 2a. Imports and data loading**
# 
# Start by importing the packages that you will need to load and explore the dataset. Make sure to use the following import statements:
# *   `import pandas as pd`
# 
# *   `import numpy as np`
# 

# In[1]:


# Import packages
### YOUR CODE HERE ###
import pandas as pd
import numpy as np


# Then, load the dataset into a dataframe. Creating a dataframe will help you conduct data manipulation, exploratory data analysis (EDA), and statistical activities.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")


# ### **Task 2b. Understand the data - Inspect the data**
# 
# View and inspect summary information about the dataframe by **coding the following:**
# 
# 1. `data.head(10)`
# 2. `data.info()`
# 3. `data.describe()`
# 
# *Consider the following questions:*
# 
# **Question 1:** When reviewing the first few rows of the dataframe, what do you observe about the data? What does each row represent?
# 
# **Question 2:** When reviewing the `data.info()` output, what do you notice about the different variables? Are there any null values? Are all of the variables numeric? Does anything else stand out?
# 
# **Question 3:** When reviewing the `data.describe()` output, what do you notice about the distributions of each variable? Are there any questionable values? Does it seem that there are outlier values?
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[3]:


# Display and examine the first ten rows of the dataframe
### YOUR CODE HERE ###
data.head(10)


# In[4]:


# Get summary info
### YOUR CODE HERE ###
data.info()


# In[5]:


# Get summary statistics
### YOUR CODE HERE ###
data.describe()


# ===> ENTER YOUR RESPONSE TO QUESTIONS 1-3 HERE
# 
# 1 - Each row into the dataframe represents a claim from a customer with the information of the video related to the claim.
# 2 - Not all columns are numeric, claim_status, video_transcription_text, verified_status and author_ban_status are object type. There are no null values.
# 3 - There are 19382 claims, but only 19084 with data in views, likes, shares, downloads and comments despite being non null. There are videos with no likes, shares, downloads or comments.

# ### **Task 2c. Understand the data - Investigate the variables**
# 
# In this phase, you will begin to investigate the variables more closely to better understand them.
# 
# You know from the project proposal that the ultimate objective is to use machine learning to classify videos as either claims or opinions. A good first step towards understanding the data might therefore be examining the `claim_status` variable. Begin by determining how many videos there are for each different claim status.

# In[6]:


# What are the different values for claim status and how many of each are in the data?
### YOUR CODE HERE ###
data['claim_status'].value_counts()


# **Question:** What do you notice about the values shown?
# There are more claims than opinions in the dataset.

# Next, examine the engagement trends associated with each different claim status.
# 
# Start by using Boolean masking to filter the data according to claim status, then calculate the mean and median view counts for each claim status.

# In[7]:


# What is the average view count of videos with "claim" status?
### YOUR CODE HERE ###
mask = data['claim_status'] == 'claim'
data[mask].mean()[['video_view_count']]


# In[8]:


# What is the average view count of videos with "opinion" status?
### YOUR CODE HERE ###
mask = data['claim_status'] == 'opinion'
data[mask].mean()[['video_view_count']]


# **Question:** What do you notice about the mean and media within each claim category? The mean view count for claim status are much higher than the mean view for opinion status.
# 
# Now, examine trends associated with the ban status of the author.
# 
# Use `groupby()` to calculate how many videos there are for each combination of categories of claim status and author ban status.

# In[18]:


# Get counts for each group combination of claim status and author ban status
### YOUR CODE HERE ###
data.groupby(['author_ban_status','claim_status']).count()[['#']]


# **Question:** What do you notice about the number of claims videos with banned authors? Why might this relationship occur?
# There are less claim videos by banned authors than from active, this occurs due to the author being banned, their videos cannot be claimed anymore.
# 
# Continue investigating engagement levels, now focusing on `author_ban_status`.
# 
# Calculate the median video share count of each author ban status.

# In[21]:


### YOUR CODE HERE ###
data.groupby(['author_ban_status']).median()[['video_share_count']]


# In[ ]:


# What's the median video share count of each author ban status?
### YOUR CODE HERE ###


# **Question:** What do you notice about the share count of banned authors, compared to that of active authors? Explore this in more depth.
# Banned authors median video share is higher than the ones active, most likely due to being shared before banned
# 
# Use `groupby()` to group the data by `author_ban_status`, then use `agg()` to get the count, mean, and median of each of the following columns:
# * `video_view_count`
# * `video_like_count`
# * `video_share_count`
# 
# Remember, the argument for the `agg()` function is a dictionary whose keys are columns. The values for each column are a list of the calculations you want to perform.

# In[22]:


### YOUR CODE HERE ###
data.groupby(['author_ban_status']).agg(['count','mean','median'])[['video_view_count','video_like_count','video_share_count']]


# **Question:** What do you notice about the number of views, likes, and shares for banned authors compared to active authors?
# There are less claims from banned videos than under review and active authors, but their  mean and median views, likes and shares are higher than the rest.
# 
# Now, create three new columns to help better understand engagement rates:
# * `likes_per_view`: represents the number of likes divided by the number of views for each video
# * `comments_per_view`: represents the number of comments divided by the number of views for each video
# * `shares_per_view`: represents the number of shares divided by the number of views for each video

# In[23]:


# Create a likes_per_view column
### YOUR CODE HERE ###
data['likes_per_view'] = data['video_like_count']/data['video_view_count']

# Create a comments_per_view column
### YOUR CODE HERE ###
data['comments_per_view'] = data['video_comment_count']/data['video_view_count']

# Create a shares_per_view column
### YOUR CODE HERE ###
data['shares_per_view'] = data['video_share_count']/data['video_view_count']


# Use `groupby()` to compile the information in each of the three newly created columns for each combination of categories of claim status and author ban status, then use `agg()` to calculate the count, the mean, and the median of each group.

# In[24]:


### YOUR CODE HERE ###
data.groupby(['author_ban_status','claim_status']).agg(['count','mean','median'])[['likes_per_view','comments_per_view','shares_per_view']]


# **Question:**
# 
# How does the data for claim videos and opinion videos compare or differ? Consider views, comments, likes, and shares.
# 
# Claims tend to be for videos with more interactions, while Opinions tend to be for videos with less interactions, the higher percentage of interactions are banned authors.

# ### **Given your efforts, what can you summarize for Rosie Mae Bradshaw and the TikTok data team?**
# 
# *Note for Learners: Your answer should address TikTok's request for a summary that covers the following points:*
# 
# *   What percentage of the data is comprised of claims and what percentage is comprised of opinions?
# *   What factors correlate with a video's claim status?
# *   What factors correlate with a video's engagement level?
# 

# ==> ENTER YOUR RESPONSE HERE
# 50.34% of the data is comprised of claims, 49.66% is comprised of opinions.
# Claims are considerably higher than opinions in banned and under review authors, contrary to active authors where opinions are higher.
# Claims are higher in videos with high engagement.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
