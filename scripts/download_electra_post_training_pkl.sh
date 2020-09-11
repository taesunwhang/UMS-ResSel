#!/usr/bin/env bash

export directory=$PWD/data/electra_post_training
if [ ! -d "$directory" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p $directory
fi

export file_name=electra_post_training.zip
if [ -f $PWD/data/electra_post_training/ubuntu_electra_post_trianing.pkl ]; then
    echo "$file_name exists"
else
    echo "$file_name does not exist"
    export file_id=1eitfjOONhTSVoWi0mgxHxvMWpmSKmGpC

    ## WGET ##
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    unzip $file_name -d data/electra_post_training/
    rm $file_name
fi
