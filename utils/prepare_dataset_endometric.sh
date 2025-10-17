#!/bin/bash

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATASET_DIR="${1}"
DATASET_NAME="${2}"

CAMERA_MODEL="kb4"


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



SEQUENCE_SF="/media/sf_shared_folder/${DATASET_NAME}"
SEQUENCE_LOCAL="${DATASET_DIR}"

cp -r $SEQUENCE_SF $SEQUENCE_LOCAL
echo "Copy files in $SEQUENCE_LOCAL"


if [ ! -d "${DATASET_DIR}/${DATASET_NAME}" ]; then
    echo "Dataset name not found at '${DATASET_DIR}'. Check the provided path."
    usage
fi

# copy required files to calibration folder
cp ${SCRIPT_DIR}/vicalib_calibration_and_poses.sh ${DATASET_DIR} || exit 2


cd ${DATASET_DIR}

PATTERN_SIZE=""
for filename in ${DATASET_NAME}/*_pattern.xml; do
  basefile=$(basename "$filename")                 # times_06_pattern.xml
  PATTERN_SIZE="${basefile%%_pattern.xml}"               # times_06
  echo "Pattern size: $basefile"
done

cp ${DATASET_NAME}/*_pattern.xml "${DATASET_NAME}/${DATASET_NAME}_pattern.xml" || exit 2
./vicalib_calibration_and_poses.sh ${DATASET_NAME} $PATTERN_SIZE ${CAMERA_MODEL} || exit 2


# clean temporal files
rm vicalib_calibration_and_poses.sh

mv "${DATASET_NAME}/${DATASET_NAME}_poses.csv" $SEQUENCE_SF

cd ${CURRENT_DIR}