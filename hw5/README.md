## DLA hw5. TTS 2

This is a WaveNet vcoder model you can train to turn melspectrograms into wavs.
The model is based on https://arxiv.org/pdf/1609.03499.pdf

To reproduce the results do:
```
chmod +x download.sh
./download.sh
pip install -r requirements.txt
# if you want to log the results use wandb:
wandb login your_key
python main.py
```

To use the trained model use the inference method on your melspectrogram

You can look into the code with some examples in dla_hw5.ipynb
