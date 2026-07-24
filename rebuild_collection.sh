#/usr/bin/env bash

# Merge the split binary files in order.
cat data/train/part_aa data/train/part_ab data/train/part_ac data/train/part_ad data/train/part_ae data/train/part_af data/train/part_ag data/train/part_ah data/train/part_ai data/train/part_aj data/train/part_ak > data/train/genIR.collection.tsv.xz
# Uncompress the file
xz -dv data/train/genIR.collection.tsv.xz
# Verify that it worked
md5sum -c collection.md5 
if [ $? -ne 0 ];
then
  echo "Collection regeneration failed. Proceed with caution."
else
  echo "Collection successfully recreated."
fi
