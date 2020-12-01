import os
from detection import detection

detection.detect("test/milo", "model/craft_mlt_25k.pth")
os.system("python3 ./deep-text-recognition-benchmark/demo.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder cropped/ --saved_model deep-text-recognition-benchmark/TPS-ResNet-BiLSTM-Attn.pth")