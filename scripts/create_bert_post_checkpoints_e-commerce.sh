#!/usr/bin/env bash
# download e-commerce_post_training.txt
export file_name=e-commerce_post_training.txt
if [ -f $PWD/data/e-commerce/$file_name ]; then
    echo "$file_name exists"
else
    echo "$file_name does not exist"
    export file_id=1-O-HBk-NPwa0AwGQqCkQ8UtGFuCRpw3u

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    mv $file_name $PWD/data/e-commerce/
fi

python3 post_train/bert/create_post_training_data.py --input_file ./data/e-commerce/e-commerce_post_training.txt --output_file ./data/e-commerce/e-commerce_post_training.hdf5 --bert_pretrained bert-base-wwm-chinese --dupe_factor 10