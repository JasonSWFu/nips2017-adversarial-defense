# nips2017-adversarial-defense

This is the FINAL defense model of team StupidHans in the NIPS 2017: Defense Against Adversarial Attack competition. 

Our model tries to apply a denoise model before our classifier. This denoise model is trained in the code ImageNet_denoise_fixed.py.
Then it is finne-tuned with the classifier in the code Keras_Xception_cascaded_finetune_training.py.
Finally, different models are ensembled to the final submitted model.


The weights and models can be downloaded from [here](https://drive.google.com/open?id=0ByZBoQo28N9BQi1TUUlBbHhZdDA)

