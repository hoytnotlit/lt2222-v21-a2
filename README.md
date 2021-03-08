# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

Your name: Minerva (gussuvmi)

Notes

Due to the design choice of not including parts of NE's in the context features, NE's with multiple words will get multiple instances. For example "hyde park" gets two instances where both instances are identical. This could be changed by skipping the next parts of the NE but I deemed this as desired functionality because each individual word of the NE should be considered in training.
            

Observations

At first I did not implement including words before and after the NE but not parts of the NE in the context and the results of my training were unsatisfactory. I ran the program through a few times and the model was predicting at most two classes, in most cases just one class.

I decided to implement the changes for context features and the predictions started working significantly better. Not all classes were still predicted, however. I would assume tweaking the data by for example removing uninformative words such as determiners would improve the model a bit more.

However I ran the code again the next day and the predictions were bad again so I do not know what happened. It seems that training the model a couple of times makes the predictions better so my assumption is this is what happened previously.

The predictions on the training data are unsurprisingly better than those for the test data. This is because the training data is biased because the model has seen it before. This results in more correct guesses when using the training data for testing purposes. More classes are being guessed for training data than for testing data.

In both cases, some classes are predicted less well than others. Classes such as "art" and "nat" have just a few test cases (test: 4 and 2 train: 65 and 20 on my try when writing this) whereas classes such as "geo" and "per" have more cases (test: 530 and 499 train: 1996 and 1866). Classes with more cases are being predicted correctly more often than those with fewer cases. Classes with more cases are also predicted as incorrect classes more, it seems. For example the class "tim" has more incorrect predictions as "geo" (test: 154 train: 619) than any other class. To compare the correct guesses for "tim" are 16 for test data and 76 for train data.