#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

DATASET_DIR="./CityScapes"
mkdir -p "${DATASET_DIR}"
cd "${DATASET_DIR}"

wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=atrium&password=heyerato1993&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1

# 1 -> gtFine_trainvaltest.zip (241MB)
# 2 -> gtCoarse.zip (1.3GB)
# 3 -> leftImg8bit_trainvaltest.zip (11GB)
# 4 -> leftImg8bit_trainextra.zip (44GB)
# 8 -> camera_trainvaltest.zip (2MB)
# 9 -> camera_trainextra.zip (8MB)
# 10 -> vehicle_trainvaltest.zip (2MB)
# 11 -> vehicle_trainextra.zip (7MB)
# 12 -> leftImg8bit_demoVideo.zip (6.6GB)
# 28 -> gtBbox_cityPersons_trainval.zip (2.2MB)
