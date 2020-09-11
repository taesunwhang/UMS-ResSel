#!/usr/bin/env bash
# download ubuntu_post_training.txt
export file_name=ubuntu_post_training.txt
if [ -f $PWD/data/ubuntu_corpus_v1/$file_name ]; then
    echo "$file_name exists"
else
    echo "$file_name does not exist"
    export file_id=1mYS_PrnrKx4zDWOPTFhx_SeEwdumYXCK

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    mv $file_name $PWD/data/ubuntu_corpus_v1/
fi

python3 post_train/bert/create_post_training_data.py --input_file ./data/ubuntu_corpus_v1/ubuntu_post_training.txt --output_file ./data/ubuntu_corpus_v1/ubuntu_post_training.hdf5 --bert_pretrained bert-base-uncased --dupe_factor 10