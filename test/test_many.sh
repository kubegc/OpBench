#!/bin/bash

# touch "./res/test_with_threads_resnet-18_gpu.txt"
# touch "./res/test_with_threads_mobilenet_gpu.txt"
# touch "./res/test_with_threads_squeezenet_v1.1_gpu.txt"
# touch "./res/test_with_threads_inception_v3_gpu.txt"

# touch "./res/test_with_threads_resnet-18_cpu.txt"
# touch "./res/test_with_threads_mobilenet_cpu.txt"
# touch "./res/test_with_threads_squeezenet_v1.1_cpu.txt"
# touch "./res/test_with_threads_inception_v3_cpu.txt"

# touch "./res/test_with_threads_resnet-18_ft-cpu.txt"
# touch "./res/test_with_threads_mobilenet_ft-cpu.txt"
# touch "./res/test_with_threads_squeezenet_v1.1_ft-cpu.txt"
# touch "./res/test_with_threads_inception_v3_ft-cpu.txt"

touch "./res/test_with_threads_bert_ft-cpu.txt"
touch "./res/test_with_threads_gpt2_ft-cpu.txt"
touch "./res/test_with_threads_roberta_ft-cpu.txt"

# for ((i = 1; i <= 40; i++)); do
#     threads_num=$i

#     alltimestr='0'
#     for ((z = 0; z < 3; z++)); do
#         output=`TVM_NUM_THREADS=$i python ./test_with_threads.py resnet-18 cpu`
#         timestr=${output#*Total_time}
#         timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
#         alltimestr+='+'
#         alltimestr+=$timeval
#     done
#     alltime='scale=3;('$alltimestr')/3'
#     avertime=`echo $alltime | bc`
#     input=$i' '$avertime
#     echo $input >> ./res/test_with_threads_resnet-18_cpu.txt
# done

# for ((i = 1; i <= 40; i++)); do
#     threads_num=$i

#     alltimestr='0'
#     for ((z = 0; z < 3; z++)); do
#         output=`TVM_NUM_THREADS=$i python ./test_with_threads.py mobilenet cpu`
#         timestr=${output#*Total_time}
#         timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
#         alltimestr+='+'
#         alltimestr+=$timeval
#     done
#     alltime='scale=3;('$alltimestr')/3'
#     avertime=`echo $alltime | bc`
#     input=$i' '$avertime
#     echo $input >> ./res/test_with_threads_mobilenet_cpu.txt
# done

# for ((i = 1; i <= 40; i++)); do
#     threads_num=$i

#     alltimestr='0'
#     for ((z = 0; z < 3; z++)); do
#         output=`TVM_NUM_THREADS=$i python ./test_with_threads.py squeezenet_v1.1 cpu`
#         timestr=${output#*Total_time}
#         timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
#         alltimestr+='+'
#         alltimestr+=$timeval
#     done
#     alltime='scale=3;('$alltimestr')/3'
#     avertime=`echo $alltime | bc`
#     input=$i' '$avertime
#     echo $input >> ./res/test_with_threads_squeezenet_v1.1_cpu.txt
# done

# for ((i = 1; i <= 40; i++)); do
#     threads_num=$i

#     alltimestr='0'
#     for ((z = 0; z < 3; z++)); do
#         output=`TVM_NUM_THREADS=$i python ./test_with_threads.py inception_v3 cpu`
#         timestr=${output#*Total_time}
#         timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
#         alltimestr+='+'
#         alltimestr+=$timeval
#     done
#     alltime='scale=3;('$alltimestr')/3'
#     avertime=`echo $alltime | bc`
#     input=$i' '$avertime
#     echo $input >> ./res/test_with_threads_inception_v3_cpu.txt
# done

# for ((i = 0; i <= 10; i++)); do
#     for ((j = 0; j <= 10; j++)); do
#         max_num_threads=$[2**$i]
#         thread_warp_size=$[2**$j]

#         alltimestr='0'
#         for ((z = 0; z < 3; z++)); do
#             output=`python ./test_with_threads.py resnet-18 gpu $max_num_threads $thread_warp_size`
#             timestr=${output#*Total_time}
#             timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
#             alltimestr+='+'
#             alltimestr+=$timeval
#         done
#         alltime='scale=3;('$alltimestr')/3'
#         avertime=`echo $alltime | bc`
#         input=$i' '$j' '$avertime
#         echo $input >> ./res/test_with_threads_resnet-18_gpu.txt
#     done
# done

# for ((i = 0; i <= 10; i++)); do
#     for ((j = 0; j <= 10; j++)); do
#         max_num_threads=$[2**$i]
#         thread_warp_size=$[2**$j]

#         alltimestr='0'
#         for ((z = 0; z < 3; z++)); do
#             output=`python ./test_with_threads.py mobilenet gpu $max_num_threads $thread_warp_size`
#             timestr=${output#*Total_time}
#             timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
#             alltimestr+='+'
#             alltimestr+=$timeval
#         done
#         alltime='scale=3;('$alltimestr')/3'
#         avertime=`echo $alltime | bc`
#         input=$i' '$j' '$avertime
#         echo $input >> ./res/test_with_threads_mobilenet_gpu.txt
#     done
# done

# for ((i = 0; i <= 10; i++)); do
#     for ((j = 0; j <= 10; j++)); do
#         max_num_threads=$[2**$i]
#         thread_warp_size=$[2**$j]

#         alltimestr='0'
#         for ((z = 0; z < 3; z++)); do
#             output=`python ./test_with_threads.py squeezenet_v1.1 gpu $max_num_threads $thread_warp_size`
#             timestr=${output#*Total_time}
#             timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
#             alltimestr+='+'
#             alltimestr+=$timeval
#         done
#         alltime='scale=3;('$alltimestr')/3'
#         avertime=`echo $alltime | bc`
#         input=$i' '$j' '$avertime
#         echo $input >> ./res/test_with_threads_squeezenet_v1.1_gpu.txt
#     done
# done

# for ((i = 0; i <= 10; i++)); do
#     for ((j = 0; j <= 10; j++)); do
#         max_num_threads=$[2**$i]
#         thread_warp_size=$[2**$j]

#         alltimestr='0'
#         for ((z = 0; z < 3; z++)); do
#             output=`python ./test_with_threads.py inception_v3 gpu $max_num_threads $thread_warp_size`
#             timestr=${output#*Total_time}
#             timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
#             alltimestr+='+'
#             alltimestr+=$timeval
#         done
#         alltime='scale=3;('$alltimestr')/3'
#         avertime=`echo $alltime | bc`
#         input=$i' '$j' '$avertime
#         echo $input >> ./res/test_with_threads_inception_v3_gpu.txt
#     done
# done

# for ((i = 1; i <= 64; i++)); do
#     threads_num=$i

#     alltimestr='0'
#     for ((z = 0; z < 3; z++)); do
#         output=`TVM_NUM_THREADS=$i python3 ./test_with_threads.py resnet-18 cpu`
#         timestr=${output#*Total_time}
#         timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
#         alltimestr+='+'
#         alltimestr+=$timeval
#     done
#     alltime='scale=3;('$alltimestr')/3'
#     avertime=`echo $alltime | bc`
#     input=$i' '$avertime
#     echo $input >> ./res/test_with_threads_resnet-18_ft-cpu.txt
# done

# for ((i = 1; i <= 64; i++)); do
#     threads_num=$i

#     alltimestr='0'
#     for ((z = 0; z < 3; z++)); do
#         output=`TVM_NUM_THREADS=$i python3 ./test_with_threads.py mobilenet cpu`
#         timestr=${output#*Total_time}
#         timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
#         alltimestr+='+'
#         alltimestr+=$timeval
#     done
#     alltime='scale=3;('$alltimestr')/3'
#     avertime=`echo $alltime | bc`
#     input=$i' '$avertime
#     echo $input >> ./res/test_with_threads_mobilenet_ft-cpu.txt
# done

# for ((i = 1; i <= 64; i++)); do
#     threads_num=$i

#     alltimestr='0'
#     for ((z = 0; z < 3; z++)); do
#         output=`TVM_NUM_THREADS=$i python3 ./test_with_threads.py squeezenet_v1.1 cpu`
#         timestr=${output#*Total_time}
#         timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
#         alltimestr+='+'
#         alltimestr+=$timeval
#     done
#     alltime='scale=3;('$alltimestr')/3'
#     avertime=`echo $alltime | bc`
#     input=$i' '$avertime
#     echo $input >> ./res/test_with_threads_squeezenet_v1.1_ft-cpu.txt
# done

# for ((i = 1; i <= 64; i++)); do
#     threads_num=$i

#     alltimestr='0'
#     for ((z = 0; z < 3; z++)); do
#         output=`TVM_NUM_THREADS=$i python3 ./test_with_threads.py inception_v3 cpu`
#         timestr=${output#*Total_time}
#         timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
#         alltimestr+='+'
#         alltimestr+=$timeval
#     done
#     alltime='scale=3;('$alltimestr')/3'
#     avertime=`echo $alltime | bc`
#     input=$i' '$avertime
#     echo $input >> ./res/test_with_threads_inception_v3_ft-cpu.txt
# done


for ((i = 1; i <= 64; i++)); do
    threads_num=$i

    alltimestr='0'
    for ((z = 0; z < 3; z++)); do
        output=`TVM_NUM_THREADS=$i python3 ./test_with_threads.py transformers bert cpu`
        timestr=${output#*Total_time}
        timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
        alltimestr+='+'
        alltimestr+=$timeval
    done
    alltime='scale=3;('$alltimestr')/3'
    avertime=`echo $alltime | bc`
    input=$i' '$avertime
    echo $input >> ./res/test_with_threads_bert_ft-cpu.txt
done

for ((i = 1; i <= 64; i++)); do
    threads_num=$i

    alltimestr='0'
    for ((z = 0; z < 3; z++)); do
        output=`TVM_NUM_THREADS=$i python3 ./test_with_threads.py transformers gpt2 cpu`
        timestr=${output#*Total_time}
        timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
        alltimestr+='+'
        alltimestr+=$timeval
    done
    alltime='scale=3;('$alltimestr')/3'
    avertime=`echo $alltime | bc`
    input=$i' '$avertime
    echo $input >> ./res/test_with_threads_gpt2_ft-cpu.txt
done

for ((i = 1; i <= 64; i++)); do
    threads_num=$i

    alltimestr='0'
    for ((z = 0; z < 3; z++)); do
        output=`TVM_NUM_THREADS=$i python3 ./test_with_threads.py transformers roberta cpu`
        timestr=${output#*Total_time}
        timeval=`echo $timestr | sed 's/-//g' | sed 's/ //g'`
        alltimestr+='+'
        alltimestr+=$timeval
    done
    alltime='scale=3;('$alltimestr')/3'
    avertime=`echo $alltime | bc`
    input=$i' '$avertime
    echo $input >> ./res/test_with_threads_roberta_ft-cpu.txt
done