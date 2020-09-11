#!/usr/bin/env bash

export file_name=ubuntu.zip
if [ -f $PWD/data/ubuntu_corpus_v1/ubuntu_train.pkl ]; then
    echo "ubuntu_train.pkl exists"
else
    echo "ubuntu_train.pkl does not exist"
    export file_id=1VKQaNNC5NR-6TwVPpxYAVZQp_nwK3d5u

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    unzip $file_name -d data/ubuntu_corpus_v1
    rm -r $file_name
fi

export file_name=douban.zip
if [ -f $PWD/data/douban/douban_train.pkl ]; then
    echo "douban_train.pkl exists"
else
    echo "douban_train.pkl does not exist"
    export file_id=1B9vuTtQW0_8PP2hISCmey2qGXnQJQEMy

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    unzip $file_name -d data/douban
    rm -r $file_name
fi

export file_name=e-commerce.zip
if [ -f $PWD/data/e-commerce/e-commerce_train.pkl ]; then
    echo "e-commerce_train.pkl exists"
else
    echo "e-commerce_train.pkl does not exist"
    export file_id=1ZuUaTw2V6pcp2QynYIdVLFtVMyGZEbIG

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    unzip $file_name -d data/e-commerce
    rm -r $file_name
fi