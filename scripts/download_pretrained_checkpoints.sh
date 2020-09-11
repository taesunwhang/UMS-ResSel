#!/usr/bin/env bash

# bert_base
export file_name=bert-base-uncased-pytorch_model.bin
if [ -f $PWD/resources/bert-base-uncased/$file_name ]; then
    echo "$file_name exists"
else
    wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
    mv bert-base-uncased-pytorch_model.bin resources/bert-base-uncased/
fi

export file_name=bert-base-chinese-pytorch_model.bin
if [ -f $PWD/resources/bert-base-chinese/$file_name ]; then
    echo "$file_name exists"
else
    wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin
    mv bert-base-chinese-pytorch_model.bin resources/bert-base-chinese/
fi

export file_name=bert-base-wwm-chinese.zip
if [ -f $PWD/resources/bert-base-wwm-chinese/bert-base-wwm-chinese-pytorch_model.bin ]; then
    echo "bert-base-wwm-chinese-pytorch_model.bin exists"
else
    echo "bert-base-wwm-chinese-pytorch_model.bin does not exist"
    export file_id=1m7XFE7r9_gBn6bud20DxkQ8V0oRZLisV

    ## WGET ##
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    unzip $file_name -d resources/bert-base-wwm-chinese
    rm -r $file_name
fi

# bert_post
# ubuntu
export file_name=bert-post-uncased-pytorch_model.pth
if [ -f $PWD/resources/bert-post-uncased/$file_name ]; then
    echo "$file_name exists"
else
    echo "$file_name does not exist"
    export file_id=1jt0RhVT9y2d4AITn84kSOk06hjIv1y49

    ## WGET ##
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    mv $file_name resources/bert-post-uncased/
fi

# douban
export file_name=bert-post-douban-pytorch_model.pth
if [ -f $PWD/resources/bert-post-douban/$file_name ]; then
    echo "$file_name exists"
else
    echo "$file_name does not exist"
    export file_id=1yqdZiVuyURACGBluKnAr0ARB-Sjbd1za

    ## WGET ##
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    mv $file_name resources/bert-post-douban/
fi

# e-commerce
export file_name=bert-post-ecommerce-pytorch_model.pth
if [ -f $PWD/resources/bert-post-ecommerce/$file_name ]; then
    echo "$file_name exists"
else
    echo "$file_name does not exist"
    export file_id=1-TFyyH2KMMZ75AKgeb8brdU8lPRFPhQf

    ## WGET ##
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    mv $file_name resources/bert-post-ecommerce/
fi


# electra_base
export file_name=electra-base.zip
if [ -f $PWD/resources/electra-base/electra-base-gen-pytorch_model.bin ]; then
    echo "electra-base-pytorch_model.bin exists"
else
    echo "electra-base-pytorch_model.bin does not exist"
    export file_id=1xbdiisYRHupIa1ztAsjxsi8JaHIrvWBV

    ## WGET ##
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    unzip $file_name -d resources/electra-base
    rm -r $file_name
fi

export file_name=electra-base-chinese.zip
if [ -f $PWD/resources/electra-base-chinese/electra-base-chinese-gen-pytorch_model.bin ]; then
    echo "electra-base-chinese-pytorch_model.bin exists"
else
    echo "electra-base-chinese-pytorch_model.bin does not exist"
    export file_id=1V6zYegAGnjJiFfhkHHv0cwph1V25YZnX

    ## WGET ##
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    unzip $file_name -d resources/electra-base-chinese
    rm -r $file_name
fi

# electra_post
export file_name=electra-post-douban-pytorch_model.pth
if [ -f $PWD/resources/electra-post-douban/$file_name ]; then
    echo "$file_name exists"
else
    echo "$file_name does not exist"
    export file_id=1DED1uUqtxIjXbYooD3cR4Ih2W1di2x8r

    ## WGET ##
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    mv $file_name resources/electra-post-douban/
fi

export file_name=electra-post-ecommerce-pytorch_model.pth
if [ -f $PWD/resources/electra-post-ecommerce/$file_name ]; then
    echo "$file_name exists"
else
    echo "$file_name does not exist"
    export file_id=1zeHhd_FRNJyoTh0gkBLlO73BqUl9H43Z

    ## WGET ##
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    mv $file_name resources/electra-post-ecommerce/
fi




export file_name=electra-post-pytorch_model.pth
if [ -f $PWD/resources/electra-post/$file_name ]; then
    echo "$file_name exists"
else
    echo "$file_name does not exist"
    export file_id=14alN7bdjLvncgrjeOhZ5ye8rJRBu5Unx

    ## WGET ##
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    mv $file_name resources/electra-post/
fi


