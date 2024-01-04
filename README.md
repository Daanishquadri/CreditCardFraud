# Credit Card Fraud Detection Using Python And Machine Learning

# Link to Code: https://github.com/Daanishquadri/CreditCardFraud.git
# Link for the data: Credit Card Fraud Detection (kaggle.com)

# ABSTRACT 

This dataset included credit card transactions performed by cardholders throughout Europe in
September 2013. This dataset shows the transaction that took place over a two-day period. Of the 284,807
transactions, out of which 492 were fraudulent. Because of the extreme imbalance in the dataset, 0.172% of
all transactions belong to the positive class(frauds). Its sole input variables are numbers that come from the
PCA transformation. Regretfully, we are unable to offer the original features and further context for the
data owing due to privacy concerns. Features V1, V2,... The primary components identified by PCA are
V28; “Time” and “Amount” are the only features that have not undergone PCA transformation. The
seconds that pass between each transaction and the dataset’s initial transaction are contained in the feature
“Time.” ‘Amount as a feature is the transaction Amount: This function can be applied to cost-sensitive,
example-dependent learning. The response variable, ‘Class’, takes a value 0 in the absence of fraud and 1 in
the case of fraud. We advise calculating the accuracy using the Area Under the Precision-Recall
Curve(AUPRC) in light of the class imbalance ratio. The accuracy of the confusion matrix has no bearing
on imbalanced classification.

# I. INTRODUCTION

The unapproved use of credit card information for financial
benefit is known as credit card fraud. Sensitive card
information is used by thieves to conduct fraudulent
transactions, which costs financial institutions and
consumers money. It is critical to identify and stop credit
card fraud in order to protect people's finances and preserve
the integrity of the financial system.
The importance of detection and prevention has a financial
component since credit card fraud may cause large financial
losses for both individuals and companies. By detecting and
preventing it early on, these losses can be reduced. When it
comes to preventing fraud and building credit card users'
trust, confidence and trust work incredibly well. Financial
firms that show a dedication to security have the potential
to gain the trust of their customers. Maintaining the
integrity of operations is essential for financial institutions.
A safe and dependable financial environment is maintained
in part by fraud prevention strategies.
The most crucial step towards improving fraud detection
and prevention is behavior monitoring by credit card users
since this is a proactive measure. This entails closely
monitoring any patterns, actions, or behaviors that might
point to fraud. Advanced analytics and machine learning
can be used to detect trends that are typically linked to
fraudulent transactions in order to detect fraud. keeping an
eye on transaction history and contrasting it with
established patterns of valid transactions. Monitoring
unwanted access attempts or other activity that can point to
a security breach is made easier with the use of intrusion
detection. by using the systems for detecting instructions to
locate and address such dangers. Payment default is a
financial risk that is mostly managed via financial risk
analysis, which uses payment behavior and history to
forecast the probability of payment default and find early
warning indicators. The user's conduct, which may differ
from the cardholder's typical activity, might be used to
identify objective behavior. Additionally, it is capable of
identifying behaviors that could cause suspicion or point to
possible fraud.
Unlawful access to your account, or unlawful entry into a
user's credit card account by third parties, is one of the
complex issues in credit card security. This can result in
unauthorized purchases, identity theft, and the
compromising of sensitive financial information. Strong
authentication procedures, multi-factor authentication, and
frequent password changes all contribute to the accounts'
security by preventing unwanted access. Unauthorized
transactions via your account occur when thieves use credit
card information that has been hacked to make illicit
purchases on several websites. In essence, this may result in
monetary losses and harm to the user's reputation across a
range of online services. Using virtual credit cards for
online transactions, creating transaction alerts, and
routinely reviewing transaction history are all ways that
consumers can improve security. A class imbalance
indicates that there are more honest transactions than
dishonest ones. When it comes to credit card transactions,
the number of genuine transactions vastly exceeds that of
fraudulent ones. In the midst of a sea of valid transactions,
traditional models may find it difficult to identify rare
fraudulent events, which could essentially result in false
negatives. The problem of class imbalance can be solved by
using sophisticated machine learning algorithms, anomaly
detection methods, and oversampling of fraudulent cases.
Modifications to the transaction specifics of your purchase,
such as changing the beneficiaries or purchase amounts, are
known as changes of transaction. Additionally, it modifies
the transaction information, which may reveal possible
fraudulent activity and cause financial disparities. Risks can
be reduced by users routinely reviewing transaction
statements, utilizing real-time alerts, and reporting
suspicious activity right away.
It is critical to identify fraudulent transactions to protect
people's finances. The timely detection of unapproved
charges is essential to reducing the effect on people's
finances and averting significant losses. Trust in financial
institutions is also greatly impacted by the capacity to
identify and resolve fraudulent activities. Institutions can
increase client confidence and trustworthiness by
showcasing their dedication to security and immediately
addressing security concerns. The significance goes beyond
stopping identity theft, as identity theft strategies can be
thwarted by early fraudulent transaction identification.
Operationally speaking, prompt fraud detection lowers
expenses and facilitates effective resource allocation,
freeing up institutions to concentrate on actual client needs.
Preventing fees for unapproved transactions not only
enhances the clientele's experience but also keeps
customers. In addition, following legal and regulatory
requirements guarantee that financial institutions meet their
duties to shield clients from unsanctioned transactions,
avoiding harm to their reputation and deterioration of their
trust. Encouraging transparency in transactions cultivates a
favorable institutional reputation, draws in a devoted
clientele, and increases customer trust.
Finding unusual transaction patterns requires a
multidimensional strategy that makes use of machine
learning and advanced analytics. Algorithms use data
analysis techniques to find patterns that deviate from
typical behavior. They then use anomaly detection to
highlight transactions that significantly deviate from
established norms. Instantaneous detection is guaranteed
via real-time monitoring, allowing prompt action to stop
such fraudulent activity. By focusing on user behavior
patterns, behavioral analysis creates a baseline for typical
activities and raises red lights for unusual transactions when
there is any divergence. The procedure also includes
stopping suspicious transactions in real-time through
automated technologies that rapidly sound alarms and stop
possibly fraudulent transactions from being completed.
Security is improved with additional transaction
verification processes including customer authentication
and multi-factor authentication. Uncertainties are addressed
by human review and confirmation, which is made possible
by manual review by fraud prevention teams and human
decision-making within automated systems. Institutions can
take proactive efforts by implementing transaction
blocking, which includes temporary restrictions on
questionable transactions. Notifying customers of suspected
transactions gives them a chance to clarify or resolve any
issues.

# II. Importing the Libraries and the dataset.

During this project, I used Python as the main programming
language, and the libraries I imported during this project are
as follows:

![image](https://github.com/Daanishquadri/CreditCardFraud/assets/84735952/743c78d0-caaa-4884-ac39-03d79774bb51)
# I. Imported libraries during the project.

Once the libraries were created some of them were getting
an error which was then fixed by installing the packages by
the following command which was pip install numpy and
etc. Once the libraries were imported I had to enter the path
of the CSV file in which I had my data and just for
reference I was unable to retrieve all the data as the file was
secured I was only able to get the range from V1 to V28
which was found out once I had the data file opened.

<img width="492" alt="image" src="https://github.com/Daanishquadri/CreditCardFraud/assets/84735952/42261826-b0ea-42ec-8d30-c2158547ceae">

# II. Attached is the dataset with the range V1 to V28.

This dataset contains transactions made by credit cards in
September 2013 by European cardholders. This dataset
presents transactions that have occurred in two days. This
specific dataset only contains numerical input variables
which, after a PCA transformation, are produced.
Regretfully, we are unable to offer the original features and
further context for the data owing to privacy concerns.
Features V1, V2,... The primary components identified by
PCA are V28; "Time" and "Amount" are the only features
that have not undergone PCA transformation. The seconds
that pass between each transaction and the dataset's initial
transaction are contained in the feature "Time." The
transaction amount is represented by the feature "Amount,"
which can be utilized for cost-sensitive learning based on
examples. The response variable, 'Class', takes a value of 0
in the absence of fraud and 1 in the case of fraud. We
advise calculating the accuracy using the Area Under the
Precision-Recall Curve (AUPRC) in light of the class
imbalance ratio. The accuracy of the confusion matrix is
meaningless for unbalanced classification. Also, before
sending in the dataset for training and testing I figured out
the dependent and the independent variables and labeled
them as ‘X’ and ‘Y’ and once that was done I went ahead
and split the dataset.


# III. Exploratory Data Analysis.

The majority of the variables in the dataset are numerical,
and they are basically the result of a Principal Component
Analysis (PCA) transformation. Principal components
analysis (PCA) is a dimensionality reduction approach that
is frequently used to reduce a set of correlated variables
into a smaller set of uncorrelated variables. The utilization
of PCA frequently enables a more effective examination of
data while maintaining its fundamental attributes.
Unfortunately, there are no more details accessible about
the dataset's origins or nature because of confidentiality
concerns. The features of the dataset, designated as V1
through V28, indicate the principal components that were
obtained by the PCA procedure, which is a significant
observation to make despite this constraint. A more
condensed representation of the data is produced by each of
these features, which each captures a mixture of the original
variables.
'Time' and 'Amount,' two characteristics, have not been
PCA transformed. These features are essential for placing
the data obtained from the PCA-transformed features in
context and most likely hold onto their original values.
'Amount' may indicate a numerical value connected to each
data point, whereas 'Time' may indicate the amount of time
that has passed since a specific occurrence or observation.
A screenshot of the dataset and its organization is included
on top of the following page for better understanding. This
graphic depiction can be a useful tool for examining the
arrangement of the dataset, the distribution of its variables,
and any obvious trends or irregularities.
# IV. Visualization and code.

![image](https://github.com/Daanishquadri/CreditCardFraud/assets/84735952/5c10c307-c4c7-4b20-85c5-44dc6121a26d)

# III. Correlation matrix for creditcard.csv


Once I had done the coding part I went ahead and plotted
the correlation matrix and then I went ahead and found out
the accuracy, total number of transactions, Number of
normal transactions, Number of fraudulent transactions,
Percentage of fraud transactions, min and max of the data
amount, data shape before and after dropping external
deciding factor (time) and removing duplicate transactions,
Accuracy score of the decision tree, F1 score and the
confusion matrix for Decision tree, Accuracy score of the
K-Nearest Neighbors, F1 score and the confusion matrix for
K-Nearest Neighbors, Accuracy score of the Logistic
Regression model, F1 score, and the confusion matrix for
Logistic Regression, Accuracy score of the Random Forest
Model, F1 score, and the confusion matrix for Random
Forest Model, Accuracy score of the XG Boost model, F1
score, and the confusion matrix for XG Boost model are
provided in the images below.

# IV. Out of the 284807 transactions there are 284315
transactions which are normal and the rest 492 transactions
are fraudulent which is 0.17%

<img width="241" alt="image" src="https://github.com/Daanishquadri/CreditCardFraud/assets/84735952/c813b741-4f7d-4b86-8083-10753367192d">

# V. Minimum and Maximum Amount of Data

<img width="240" alt="image" src="https://github.com/Daanishquadri/CreditCardFraud/assets/84735952/f37780f6-4222-49ae-af61-c8b68f55647d">

# VI. Data shape before and after dropping external deciding
factor(time) and removing duplicate transactions.

<img width="241" alt="image" src="https://github.com/Daanishquadri/CreditCardFraud/assets/84735952/ac55e2da-ea27-453a-9c50-0e6666f1c194">

# VII. Decision Tree Model Data

<img width="241" alt="image" src="https://github.com/Daanishquadri/CreditCardFraud/assets/84735952/8b4e383f-07a3-4a20-9802-f25577d16fcb">

# VIII. K-Nearest Neighbor Model Data

<img width="241" alt="image" src="https://github.com/Daanishquadri/CreditCardFraud/assets/84735952/c9256a6b-e6cc-484c-a2ac-44604ddac866">

# IX. Logistic Regression Model.

<img width="241" alt="image" src="https://github.com/Daanishquadri/CreditCardFraud/assets/84735952/2e7fb4f6-861f-42a1-8fb1-e0bf9d345681">

# X. Random Forest Model

<img width="241" alt="image" src="https://github.com/Daanishquadri/CreditCardFraud/assets/84735952/41da00bf-2fc7-4ef3-9432-b3b2c696f578">

# XI. XG Boost Model.

<img width="241" alt="image" src="https://github.com/Daanishquadri/CreditCardFraud/assets/84735952/f99087c8-33a1-4392-ab53-eeabd9cc6e13">



# V. Results

<img width="320" alt="Screenshot 2024-01-04 at 11 06 09 AM" src="https://github.com/Daanishquadri/CreditCardFraud/assets/84735952/f4076c7c-64a2-412a-969f-100ecfe9ac72">

# VI. Conclusion

In conclusion We used a variety of machine learning models, including the XGBoost, K-Nearest Neighbors, Logistic Regression, Decision Tree, Support Vector, and Random Forest models, in our thorough investigation of credit card fraud detection. The most successful of these was the XGBoost model, which attained an amazing accuracy rate of 99.95%. Because our dataset is balanced, this accuracy rate—which is clearly the highest among the six models—is really remarkable. It is important to emphasize that, considering the balanced distribution of data towards one class, the achieved accuracy is relatively predicted. Attaining a high accuracy score can be misleading in fraud detection circumstances, since the incidence of fraudulent transactions is generally far smaller than that of genuine ones. In order to investigate our models' performance further, we looked at the confusion matrix. We found that, fortunately, our XGBoost model is not overfit, which suggests that it is a reliable and broadly applicable model. However, I focused on a more informative statistic—the Area Under the Precision-Recall Curve (AUPRC)—after realizing the drawbacks of utilizing accuracy as the only evaluation metric in unbalanced categorization settings. This choice was motivated by the class imbalance ratio since AUPRC offers a more sophisticated knowledge of model performance, especially when addressing uncommon occurrences like fraud. Even if the XGBoost model performs well in terms of accuracy and resilience, it is important to recognize the change that was performed on the training set. The Principal Component Analysis (PCA) transformation of the features used for model training suggests that the model is operating with a condensed version of the original variables. Although this increases productivity and lowers computing complexity, it is crucial to understand the outcomes in light of this change.In conclusion, XGBoost has been identified as an exceptional performer in credit card fraud detection as a result of our thorough review approach, which included a variety of models and evaluation measures. The comprehension of our model's effectiveness in tackling the problems presented by imbalanced datasets in fraud detection situations is enriched by the insights obtained from the confusion matrix, the class imbalance consideration, and the use of AUPRC.



# VII. References

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon

Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Aël; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. Scarff: a scalable framework for streaming credit card fraud detection with Spark, Information fusion,41, 182-194,2018,Elsevier

Carcillo, Fabrizio; Le Borgne, Yann-Aël; Caelen, Olivier; Bontempi, Gianluca. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization, International Journal of Data Science and Analytics, 5,4,285-300,2018,Springer International Publishing

Bertrand Lebichot, Yann-Aël Le Borgne, Liyun He, Frederic Oblé, Gianluca Bontempi Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection, INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, pp 78-88, 2019

Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen, Frederic Oblé, Gianluca Bontempi Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection Information Sciences, 2019

Yann-Aël Le Borgne, Gianluca Bontempi Reproducible machine Learning for Credit Card Fraud Detection - Practical Handbook

Bertrand Lebichot, Gianmarco Paldino, Wissam Siblini, Liyun He, Frederic Oblé, Gianluca Bontempi Incremental learning strategies for credit cards fraud detection, IInternational Journal of Data Science and Analytics


