The model is described in the following paper:
https://arxiv.org/abs/1806.07713

# albacore
The Albacore Clickbait Detector

1. Please create a folder with the name "data" and extract the Clickbait Challenge dataset in it. 
2. Download glove embedding vectors from http://nlp.stanford.edu/data/glove.6B.zip and put them in the data folder. Your data folder should look like the following picture.

![Alt text](1.png?raw=true "Title")

3. Put the available codes beside the data folder.

4. MyModelTraining2optimisey.py: This file will create bi-directional GRU models with different values for hyperparameters and store the accuracy of each model.

5. MyModelTraining2.py: this code will create a bi-directional GRU model with the best value for its hyperparameters in terms of mean squared error and train it on all the available data.
