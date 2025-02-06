echo "run all versions of stitching"
echo "deleting previous result images"
rm result_images/*

if [ ! -f logs.txt ]; then
    echo "delete previous logs"
    rm logs.txt
fi


echo "running simple.py"
echo "simple.py" | cat logs.txt
{ time python3 simple.py; } 2>&1 | cat logs.txt

# echo "running manual_stitching.py"
# echo "manual_stitching.py" >> logs.txt
# (time python3 manual_stitching.py) 2> logs.txt

# echo "running simple_sequential.py"
# echo "simple_sequential.py" >> logs.txt
# (time python3 simple_sequential.py) 2> logs.txt

# echo "running detailed.py"
# echo "detailed.py" >> logs.txt
# (time python3 detailed.py) 2> logs.txt

echo "done!"