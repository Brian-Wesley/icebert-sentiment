# icebert-sentiment

A model to analyze the sentiment of Icelandic texts. Easy to train on PC.

To use the latest version (v1.1) directly, please download the full model. 
Visit https://www.modelscope.cn/models/BrianWesley/icebert-sentiment-finetuned
and find icebert-sentiment-finetuned.zip. Then unzip the file.
Afterwards, run simple_icebert_sentiment.py in this repository. Inside is a simple user interface.

If you want to train it yourself, there would be no need to download the model from ModelScope.
Run SentimentAnalysis.py first, and then run ContinueLearning.py.
Adjust the settings according to your environment.
Both CSV files are needed.
Run simple_icebert_sentiment.py to interact.

In both cases should all files be in the same path.
If not, changes should be made correspondingly in the code.
Also notice that there are dependencies, which is noted in the beginning of the code.
You need to install necessary packages to run it normally.

The data used for training is from mbl.is, a famous news website in Iceland.

Based on the evaluation code (evaluate_icebert_sentiment.py), its accuracy is more than 80%, making it closer to real life production.

Further improvement upcoming...
