#!/bin/bash
DATADIR="speech_commands"
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz -O speech_commands_v0.01.tar.gz
# alternative url: https://www.dropbox.com/s/j95n278g48bcbta/speech_commands_v0.01.tar.gz?dl=1
mkdir $DATADIR && tar -C $DATADIR -xvzf speech_commands_v0.01.tar.gz 1> log
rm speech_commands_v0.01.tar.gz