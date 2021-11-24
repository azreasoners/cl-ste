# Set the number of runs for each experiment
runs=1

echo -e "========== b_p(x)+iSTE Experiments =========="
echo -e "\n\n========== mnistAdd with batch size 1 ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain mnistAddcnf --epochs 1 --hyper 40 1"
    python main.py --domain mnistAddcnf --epochs 1 --hyper 40 1
done

echo -e "\n\n========== mnistAdd with batch size 16 ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain mnistAddcnf --epochs 1 --batchSize 16 --hyper 3 0.4"
    python main.py --domain mnistAddcnf --epochs 1 --batchSize 16 --hyper 3 0.4
done

echo -e "\n\n========== mnistAdd2 with batch size 1 ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain mnistAdd2cnf --epochs 1 --hyper 40 1"
    python main.py --domain mnistAdd2cnf --epochs 1 --hyper 40 1
done

echo -e "\n\n========== add2x2 ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain add2x2cnf --numData 3000"
    python main.py --domain add2x2cnf --numData 3000
done

echo -e "\n\n========== apply2x2 ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain apply2x2cnf --numData 3000"
    python main.py --domain apply2x2cnf --numData 3000
done

echo -e "\n\n========== member(3) ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain member3cnf --numData 3000"
    python main.py --domain member3cnf --numData 3000
done

echo -e "\n\n========== member(5) ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain member5cnf --numData 3000"
    python main.py --domain member5cnf --numData 3000
done



echo -e "\n\n========== b(x)+iSTE Experiments =========="
echo -e "\n\n========== add2x2 ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain add2x2cnf --numData 3000 --b Bi --hyper 1 5"
    python main.py --domain add2x2cnf --numData 3000 --b Bi --hyper 1 5
done

echo -e "\n\n========== apply2x2 ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain apply2x2cnf --numData 3000 --b Bi"
    python main.py --domain apply2x2cnf --numData 3000 --b Bi
done

echo -e "\n\n========== member(3) ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain member3cnf --numData 3000 --b Bi --hyper 0.1 0.1"
    python main.py --domain member3cnf --numData 3000 --b Bi --hyper 0.1 0.1
done

echo -e "\n\n========== member(5) ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain member5cnf --numData 3000 --b Bi --epochs 10 --hyper 1 0.4"
    python main.py --domain member5cnf --numData 3000 --b Bi --epochs 10 --hyper 1 0.4
done



echo -e "\n\n========== b(x)+sSTE Experiments =========="
echo -e "\n\n========== add2x2 ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain add2x2cnf --numData 3000 --b Bs --hyper 1 5"
    python main.py --domain add2x2cnf --numData 3000 --b Bs --hyper 1 5
done

echo -e "\n\n========== apply2x2 ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain apply2x2cnf --numData 3000 --b Bs"
    python main.py --domain apply2x2cnf --numData 3000 --b Bs
done

echo -e "\n\n========== member(3) ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain member3cnf --numData 3000 --b Bs --hyper 0.1 0.1"
    python main.py --domain member3cnf --numData 3000 --b Bs --hyper 0.1 0.1
done

echo -e "\n\n========== member(5) ==========\n\n"
for i in $(seq 1 $runs); do 
    echo "python main.py --domain member5cnf --numData 3000 --b Bs --epochs 10 --hyper 1 0.4"
    python main.py --domain member5cnf --numData 3000 --b Bs --epochs 10 --hyper 1 0.4
done
