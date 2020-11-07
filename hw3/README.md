This is a KWS model you can train to spot the name of your favourite assistant.
The model is based on https://www.dropbox.com/s/22ah2ba7dug6pzw/KWS_Attention.pdf

To reproduce the results do:
chmod +x download.sh
./download.sh
pip install -r requirements.txt
if you want to log the results use wandb:
wandb login your_key
python main.py datapath model_checkpoint_path

If you wand to use the pretrained model, you can load it from checkpoint.pt
To see how the model performes on your audio fork the repo and do
python run your_filepath model_checkpoint_path
