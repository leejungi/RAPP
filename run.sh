
ord=0
sap=0
nap=0
num=10
declare -f ord
for i in 0 1 2 3 4 5 6 7 8 9
do
	score=$(python3 main.py --normal_class $i --epoch 200 --loss_fn 'mean')
	echo $i":"$score
	score=($score)
	ord=$(echo "$ord + ${score[0]}" | bc)
	sap=$(echo "$sap + ${score[1]}" | bc)
	nap=$(echo "$nap + ${score[2]}" | bc)

done
ord=$(echo "scale=3;$ord / 10.0" | bc)
sap=$(echo "scale=3;$sap / 10.0" | bc)
nap=$(echo "scale=3;$nap / 10.0" | bc)

echo "Result"
echo "Ord:"$ord
echo "Sap:"$sap
echo "Nap:"$nap

