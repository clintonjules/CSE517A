## Milestone 3
[Kaggle Competition Link](https://www.kaggle.com/t/106614869ce1437aaa2a0e27bcf85eb5)

Throughout the Final Project assignments, you will be working in-depth with a modified version of a real-life dataset. The modified dataset describes a control problem, namely the task of controlling a robot. There are 40 features (continuous) that describe the status of the robot, and the target value (continuous) describes the appropriate action corresponding to the feature. The training dataset which we will provide you has 8250 samples. 

The final project will be comprised of three milestones and a final project report. In Milestone 1 (due 11/18), you will do basic data exploration and compare a few baseline models. In Milestone 2 (due 12/2), you will work on data preprocessing to improve model performance. In Milestone 3 (due 12/9), you will choose a specific model and tune its hyperparameters to successfully train your own machine learning model and compete with other students in a classroom Kaggle competition. Each milestone will have a corresponding report you must submit, and these reports comprise your final project report. 

For the Kaggle competition portion of Milestone 3, you are given a test dataset on which you'll be tested. You are free to use whatever tools you have, including but not limited to, the things you learned in class, the insights you gathered from previous Milestones, and outside resources on implementing effective ML! 

Please note that while outside resources may be used, it's not recommended that you use a tool that automates the data pre-processing and model testing (ex. FLAML), as the reasonings for these decisions is required for your report.

#### Competition
Your performance will be evaluated according to your predictions on the test data. We will provide the entire test dataset but with the target column omitted. You must train your machine learning model on the training set you have been working on to predict the targets on the test data and submit your predictions.


#### Grading
As mentioned, Milestone 3 is worth 20 points, consisting of:
- 8 points for code/kernel and report
- 6 points for beating baseline 1
- 6 points for beating baseline 2
- 2 (bonus) points for first, second, and third place submissions

The 8 points for code/kernel and report rely mostly on your written report, but a small part of it is the documentation you provide in your code. 

#### Metric
The evaluation metric for this competition is [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation). The RMSE, commonly used for regression tasks, is the square root of the mean-squared error. 

##### Baseline
Baseline 1: 0.2000
Baseline 2: 0.1700 

#### Public vs. Private Scores
When you submit your predictions, you'll have a public score based on a subset of the test data (50%) and a private score based on the rest. You'll be able to see your public score as soon as you submit a prediction, but the private score will only be revealed after the deadline. The private score will determine your final performance in this competition. 

##### Submission Format
Submission files should contain two columns: "Id" and "target", and the file should contain a header and have the following format:


Id,target
1,0.1

 

Note that due to Kaggle's own limitations, you are limited to 20 submissions each day (UTC). You may choose which submission to be evaluated. Otherwise, the system will simply choose the best one.


##### Code/Notebook
To verify your work, you must create a notebook on Kaggle that generated your submission. Be sure to share it with the following usernames for grading: cynthiazma

#### Report
For Milestone 3, there is no set of instructions you have to follow. However, your report must describe the following:
1. Which feature engineering and preprocessing steps you took and why.
2. Which model/machine learning algorithm you chose and why.
3. How you optimized the hyperparameters of your machine learning algorithm.

You may include any further information that you would like to explain. Furthermore, in your code (notebook/kernel), you must provide brief documentation so that we can refer to your report to know more about your choices. This documentation doesn't have to be extensive, we just want to see which ideas you implemented, how you implemented them, and why through the code and the notebook.

