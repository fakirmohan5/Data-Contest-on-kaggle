# Data-Contest-on-kaggle

Link for the competition page : https://www.kaggle.com/c/prml-data-contest-jul-2021-rb-section

LIBRARIES: We have disallowed some libraries that abstract away complexities of the contest. We wish for you to learn and gain intuition by working with basic methods already implemented in scipy, sklearn, numpy etc. to assemble working algorithms. 

ALLOWED LIBS:  You can use 

    numpy, scipy, 
    scikit-learn, 
    pandas, csv

NOT ALLOWED LIBS: You cannot use

    Neural Networks, PyTorch, Tensorflow or any GPU intense package, since they are not within scope of our course 
    turicreate 
    surprise & LightFM - since they will abstract away most details.

Please avoid spending your time looking for more complex libraries and work with scikit-learn and numpy etc. to recreate a working algorithm.


HYPER PARAMETERS:
If your models involve hyperparameters, you may use tools for tuning them, but remember to remove (or comment-out) the hyperparameter-tuning part of the code before submitting. 

    The final code you submit should just use the right hyperparameters found previously. 


DAILY LIMITS:
To prevent brute-force, currently the number of daily submissions is capped to 2. It will be increased later. 

(IMPORTANT!) EXPECTED FILE OUTPUT:  Your code should run with a simple command 

python ROLL1_ROLL2.py

Output one .csv file as output in expected format. Please make sure to delete intermediary files your code produces (if any). Your code will be run by TAs again during evaluation. 
