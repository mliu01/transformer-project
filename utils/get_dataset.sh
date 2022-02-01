#!/bin/bash
set -e
# get dataset file from site
wget "https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc/germeval2019t1-public-data-final.zip"
unzip germeval2019t1-public-data-final.zip -d ../data
rm germeval2019t1-public-data-final.zip

#find all files and directories except train and test dataset
FILES_TO_DELETE=$(find ../data/* -not -name "blurbs_train.txt" -not -name "blurbs_test.txt" -not -name "hierarchy.txt")

echo
echo "Deleting $FILES_TO_DELETE"
read -p "Want to continue? (y/n) " -n 1 -r
echo 
if [[ $REPLY =~ ^[Yy]$ ]]
then
    #delete all files and directories except train and test dataset
    find ../data/* -not -name "blurbs_train.txt" -not -name "blurbs_test.txt" -not -name "hierarchy.txt" -delete
    echo "..deleted"
    
    #execute python script to get datasets as json files
    echo "..executing python script"
    python ../utils/dataset.py

    FILES_TO_DELETE=$(find ../data/* -not -name "blurbs_train.json" -not -name "blurbs_test.json")
    echo "Deleting $FILES_TO_DELETE"
    read -p "Continue? (y/n) " -n 1 -r
    echo 
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "..deleting"
        find ../data/* -not -name "blurbs_train.json" -not -name "blurbs_test.json" -delete
    fi
    echo "..done"
fi
echo "..exit"