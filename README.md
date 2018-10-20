# SER
Speech Emotion Recognition.

Research for Developing a Deep Learning based model for Speech Emotion Recognition. Following was the research that I conducted:

1. http://cs229.stanford.edu/proj2007/ShahHewlett%20-%20Emotion%20Detection%20from%20Speech.pdf . This paper uses SVM across 15 emotions with accuracy as low as 65% and as high as 99%. It uses acoustic features to base their classification algorithm upon. The data set used was Data Consortium’s study on Emotional Prosody and Speech Transcripts, unfortunately I was unable to get access to this data set. The reason we dropped this paper is that they used binary classification, meaning that they would compare two emotions and then decide which is which.

2. https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/IS140441.pdf . This paper did segment level classification of audio across 5 emotion classes. It uses the IEMOCAP database, but again I didn't get the data set. This paper uses vanilla DNNs for classification on acoustic features. The reason why we dropped this paper was because it was vague on telling us what features they used, and the accuracy wasn't worth it. 

3. http://www.apsipa.org/proceedings_2016/HTML/paper2016/137.pdf . This paper used 2D CNNs to classify spectrograms of 7 different emotions. It uses emo-DB as its database, we requested that data set as well but once again we didn't get the data set. 

4. https://arxiv.org/ftp/arxiv/papers/1707/1707.09917.pdf . This method also uses 2D CNNs for classification of spectrograms of audio of emotions across 7 classes. The database used is the IEMOCAP data set. The method promised 99% accuracy. We couldn’t implement this method because I was unable to implement DAARP Algorithm. 

At the end we decided to use 2D Convolution Neural Networks and try it on SAVEE dataset. Our model gave around 63% accuracy for male voices and less than 40% for female. This was due to lack of female voices in the dataset. We tried the same method on 1D CNNs and the result was almost the same. 

The code given in this repo is of 2D CNNs and RNN. 
