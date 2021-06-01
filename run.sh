epoch=200
loss_fn='sum'
relu=0.2
start_index=0
end_index=0

echo "Epoch: "$epoch
echo "Loss function: "$loss_fn
echo "Leaky Relu: "$relu
echo "Start index: "$start_index
echo "end index: "$end_index

ord=0
sap=0
nap=0

num=10

declare -f ord
declare -f sap
declare -f nap
#for i in 0 1 2 3 4 5 6 7 8 9
for i in 5 3 0 1 2 4 6 7 8 9
do
	score=$(python3 main.py --normal_class $i --epoch $epoch --loss_fn $loss_fn --relu $relu --start_index $start_index --end_index $end_index)
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

