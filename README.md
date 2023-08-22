# Species-Identification
Species can be precisely identified using **DNA barcoding and Camera Trap Images**, which helps with biodiversity monitoring and preservation. This knowledge is essential for assessing ecosystem health, identifying threatened species, and developing successful conservation plans.

# How we built it
* We first collected the relevant data from online resources.
* Then we cleaned the data for further processing.
* Then we built and trained a variety of Machine Learning models such as SVM, Logistic Regression Model, Naive Bayes etc and compared their accuracies.
* As the **Logistic Regression Model** produced the highest accuracy, we utilize it to accurately predict the species from a given DNA sequence. 
* We also use **Convolutional Neural Networks** to accuratley predict species from given camera trap images.
* We were able to achieve an **accuracy of 90.59%** with our Logistic Regression model **and 98.4%** with our Convolutional Neural Network.
* Our web application **Species Identifier** can input a DNA sequence or a camera trap image from the user and analyze it to provide species predictions along with their confidence scores. The website also contains informative articles and information regarding species preservation and conservation efforts. 
