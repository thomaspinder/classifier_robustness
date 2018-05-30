# Testing the Robustness of Classifiers in Text Classification

This repo contains code used to test the robustness of a classifier in text classification. This test is carried out by running identical classifiers on first a clean dataset, then a dataset containing varying levels of synthetic *noise*.

### Install
To avoid version clashes, it is probably best to run this code via a virtual environment and install the necessary dependencies via `pip install -r requirements.txt`.

### Running
To run the code, simply run `python3.6 main.py` from the terminal. Results will begin to populate the console output. **Add developments to this here. Perhapes a 0/1 for noise or not sysarg**.

### Data Collection
Data is scraped by artist from [Music Library Database](http://www.mldb.org/), the code for which can be found [here](www.google.com).

Once scraped, the data is stored in dictionaries and pickled for later use (pickled files are contained [here](https://github.com/tpin3694/song_lyrics_scraper/tree/master/data)).

### Noise Creation
Noise is add through three mediums: deletion, replacement and addition. One of these three methods is applied to a specified proportion of the tokens within document being classified. The exact method applied to a single token is random, ensuring that an even distribution of noise is created.

For replacement, we try and best replicate *real errors* that would come about from typos. We achieve this by replacing individiual token characters with characters very close on a standard qwerty keyboard - [code here](www.google.com).

### Classification
Classification is done through a series of metods, namely logistic regression, random forests, CNNs and LSTMs. The file containing the clasifciation methods and objects can be found [here](www.google.com).