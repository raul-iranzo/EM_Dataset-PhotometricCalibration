#!/bin/bash

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATASET_DIR=$1
DATASET_NAME="${2}"

SEQUENCE_SF="/media/sf_shared_folder/${DATASET_NAME}"




function usage()
{
    echo "Usage: ./$( basename $0 ) <DATASET_DIR> <DATASET_NAME>"
    exit 1
}

if [ "$#" -ne 2 ]; then
    echo 'Invalid number of arguments'
    usage
fi

if [ ! -d "${DATASET_DIR}" ]; then
    echo "'${DATASET_DIR}' is not a valid directory. Check the provided path."
    usage
fi

if [ ! -d "${DATASET_DIR}/${DATASET_NAME}" ]; then
    echo "Dataset name not found at '${DATASET_DIR}'. Check the provided path."
    usage
fi



# calibrate photometry 
python ${SCRIPT_DIR}/../calibration/test_hculb.py -p ${DATASET_DIR} -s ${DATASET_NAME}

cp "${DATASET_DIR}/${DATASET_NAME}/${DATASET_NAME}_photometrical.xml" $SEQUENCE_SF

# collect all output data to a CSV file
# for i in c*_*_video; do echo ${i},$(cat ${i}/${i}_output.txt | rev | cut -d' ' -f1 | rev | tr '\n' ','); done > calibrations_output.csv
