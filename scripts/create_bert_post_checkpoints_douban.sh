#!/usr/bin/env bash
# download douban_post_training.txt
export file_name=douban_post_training.txt
if [ -f $PWD/data/douban/$file_name ]; then
    echo "$file_name exists"
else
    echo "$file_name does not exist"
    export file_id=1-TwjDo8j45ZAVeWOTMvah0IBvCSuaWuR

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    mv $file_name $PWD/data/douban/
fi

python3 post_train/bert/create_post_training_data.py --input_file ./data/douban/douban_post_training.txt --output_file ./data/douban/douban_post_training.hdf5 --bert_pretrained bert-base-wwm-chinese --dupe_factor 10