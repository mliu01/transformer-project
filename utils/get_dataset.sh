#!/bin/bash
set -e
# get dataset file from site
URL="https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc/germeval2019t1-public-data-final.zip"
FILE="germeval2019t1-public-data-final.zip"

if test -f "$FILE"; then
    echo "File already exists"
else
    if wget "$URL"; then
        echo "..using wget"
    else curl -k -L -s --compressed $URL > $FILE
        echo "..using curl"
    fi
fi
unzip $FILE -d data
rm $FILE
    
#find all files and directories except train and test dataset
files () { find data/* -not -name "blurbs_train.txt" -not -name "blurbs_test.txt" -not -name "blurbs_dev.txt" -not -name "hierarchy.txt" -not -name "*.json" -not -name "*.pkl" "$@"; }

echo
echo "Deleting $(files)"
read -p "Want to continue? (y/n) " -n 1 -r
echo 
if [[ $REPLY =~ ^[Yy]$ ]]
then
    #delete all files and directories except train and test dataset
    files -delete
    echo "..deleted"
    
    #execute python script to get datasets as json files
    echo "..executing dataset script"
    python utils/dataset.py

    echo "..executing building tree script"
    python utils/build_tree_from_dataset.py

    files () { find data/* -name "*.txt" "$@"; }
    
    echo "Deleting $(files)"
    read -p "Continue? (y/n) " -n 1 -r
    echo 
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "..deleting"
        files -delete
    fi
    echo "..done"
fi
echo "..exit"