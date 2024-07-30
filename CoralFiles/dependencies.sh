#!/bin/bash

pip3 install joblib
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential gfortran libatlas-base-dev
sudo pip3 install --upgrade pip setuptools wheel
sudo pip3 install numpy

sudo pip3 install scikit-learn


# Create the directory if it doesn't exist
mkdir -p ~/bin

# Create and populate the coral-inference.sh file
cat << 'EOF' > ~/bin/coral-inference.sh
#!/bin/bash
# Shell script to run the realTimeInference.py script

# Define the path to the Python script
SCRIPT_PATH="/home/mendel/CoralBoard/CoralFiles/realTimeInference.py"

# Run the Python script with sudo
sudo /usr/bin/python3 "$SCRIPT_PATH"
EOF

chmod +x ~/bin/coral-inference.sh
echo 'export PATH=$PATH:~/bin' >> ~/.bashrc
source ~/.bashrc

# Create and populate the coral-inference.sh file
cat << 'EOF' > ~/bin/reboot.sh
#!/bin/bash
# Shell script to run the realTimeInference.py script

# Define the path to the Python script
SCRIPT_PATH="/home/mendel/CoralBoard/EnviroTesting/Reboot.py"

# Run the Python script with sudo
sudo /usr/bin/python3 "$SCRIPT_PATH"
EOF

chmod +x ~/bin/reboot.sh
echo 'export PATH=$PATH:~/bin' >> ~/.bashrc
source ~/.bashrc