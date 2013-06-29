#!/bin/zsh -e
# Usage: deduplicate_20news.zsh input_corpus_dir target_duplicates_file target_corpus_dir
#
# Create a new corpus in target_corpus_dir with duplicate text removed within
# each newsgroup in input_corpus_dir.  Duplicate lines are blanked, not deleted.
# Does not modify the input corpus.  Will override target_duplicates_file with
# the list of duplicates found.
#
# Depends on sim_text (part of sim, provided by Debian package
# 'similarity-tester').

inputdir=$1
dupsfile=$2
outdir=$3
oldpwd=`pwd`

if [ -f $dupsfile ]
then
  rm $dupsfile
fi

cd $inputdir

for newsgroup in *
  sim_text -Tn $newsgroup/* >> $dupsfile

if [ -d $outdir ] || [ -f $outdir ]
then
  echo "Error: Output directory $outdir already exists."
  exit 2
fi

cp -r $inputdir $outdir
cd $oldpwd
python deduplicate_20news.py $dupsfile $outdir
