if [ $# -ne 1 ]; then
  echo Please choose a file containing H, S, and T
  exit
fi
file=$1
if [ ! -f $file ]; then
  echo $file doens\'t exist.
  exit
fi
dir=$(dirname $file)
echo Extracting src...
grep '^S' $file | sed 's/S-//g' | sort -k 1 -n -t ' ' | awk -F'\t' '{print $2}'  > $dir/src.txt
echo Extracting hyp...
grep '^H' $file | sed 's/H-//g' | sort -k 1 -n -t ' ' | awk -F'\t' '{print $3}'  > $dir/hyp.txt
echo Extracting ref...
grep '^T' $file | sed 's/T-//g' | sort -k 1 -n -t ' ' | awk -F'\t' '{print $2}'  > $dir/trg.txt
