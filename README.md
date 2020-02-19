# Automatic-Readability-Assessment-for-Stories

I have searched and read about the concept  of readability in many papers it took me a lot of time and finally I decided to go with this 
paper :
https://www.aclweb.org/anthology/W18-0535.pdf 

In this, I have taken help from the "OneStopEnglishCorpus" dataset. The dataset contains 3 folders with the 3 different levels 
respectively  i.e. Elementary Level, Intermediate Level, and Advanced Level and all the folders contains text file based on their 
readability level.

I have extracted around 22 features for each text file which is explained inside the code in comments. And finally decided to train the 
dataset with the most powerful Support Vector Machines(SVM linear kernal). After training with this dataset the predictions are made on 
the Story text files and the final classes are stored in 'final_predictions.csv' file.

Rest is explained in the 'Documentation.txt' file.
