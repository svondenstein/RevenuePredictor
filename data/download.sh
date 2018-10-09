#!/bin/bash

###################################################################
# Script Name	:	download.sh
#
# Description	:	Downloads TGS Salt Identification Challenge
#			dataset.
# 
# Arguments	:	None
#
# Author	:	Stephen Vondenstein
# E-mail	:	stephen@vondenstein.com
# Date		:	10/08/2018
###################################################################

# Verify Kaggle installation
verifyKaggle() {
	kaggleInstall=$(which kaggle)
	if [ -z "$kaggleInstall" ]; then
		echo "Kaggle api not found. Please install the Kaggle api using `pip install kaggle` and try again"
		exit 1
	else
		verifyCredentials
	fi
}
# Verify Kaggle credentials
verifyCredentials() {
	kaggleCreds=$(kaggle config view | grep -iE 'username:\s*None')
	if [[ -z "$kaggleCreds" ]]; then
		downloadDataset
	else
		echo "Kaggle credentials improperly configured. Please go to kaggle.com/<your-username>/account and create a new API token, then save the file to ~/.kaggle/kaggle.json and try again."
		exit 1
	fi
}
# Download dataset
downloadDataset() {
	echo "Downloading dataset..."
	kaggle competitions download -c tgs-salt-identification-challenge &> /dev/null
	prepareDataset
}
# Unzip and prepare dataset
prepareDataset() {
	echo "Organizing dataset..."
	mv "./sample_submission.csv" "../submissions/"
	unzip "test.zip" &> /dev/null
	mkdir "./test"
	mv "./images/" "./test/"
	unzip "train.zip" &> /dev/null
	mkdir "./train/"
	mv -t "./train/" "./images/" "./masks/"
	rm *.zip
	echo "Done!"
}
# Main
setup(){
	verifyKaggle
}

setup
