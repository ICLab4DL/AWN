Firstly download the MIT BIH Arrhythmia dataset from https://www.physionet.org/content/mitdb/1.0.0/, named as "mit-bih-arrhythmia-database-1.0.0"
Necessary packages can be installed using "settings and ECG picturing.ipynb"
Use health.py to transform and save the data
Then execute train.sh to train the model
Performance metrics can be calculated by confusion matrix.ipynb
Parameter can be adjusted in util.py
AWN model can be modified in model/FFNN.py