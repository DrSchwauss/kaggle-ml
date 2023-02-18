echo Activating Conda env kaggle-ml
conda activate kaggle-ml
echo Downloading Titanic Dataset...
kaggle competitions download -c titanic
unzip titanic.zip 
rm titanic.zip