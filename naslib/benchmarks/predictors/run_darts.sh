predictors=(
    lce 
    lce_m 
    lcsvr
    omni_ngb
    omni_seminas
    bananas
    bonas
    gcn
    mlp
    nao
    seminas
    lgb
    ngb
    rf
    xgb
    bayes_lin_reg
    bohamiann
    dngo
    gp
    sparse_gp
    var_sparse_gp
    tabpfn_8
    tabpfn_32
    tabpfn_64
    tabpfn_128
)

experiment_types=(
    vary_train_size 
    vary_train_size 
    vary_train_size 
    vary_train_size 
    vary_train_size 
    vary_train_size 
    vary_train_size 
    vary_train_size 
    vary_train_size 
    vary_train_size 
    vary_train_size
    vary_train_size 
    vary_train_size 
    vary_train_size 
    vary_train_size
    vary_train_size 
    vary_train_size 
    vary_train_size
    vary_train_size 
    vary_train_size 
    vary_train_size 
    vary_train_size 
    vary_train_size 
    vary_train_size 
    vary_train_size
)

start_seed=$1
if [ -z "$start_seed" ]; then
    start_seed=0
fi

# folders:
base_file=./naslib
s3_folder=p301
out_dir=$s3_folder\_$start_seed

# search space / data:
search_space=darts
datasets=(cifar10)

# other variables:
trials=1
end_seed=$(($start_seed + $trials - 1))
test_size=1000

echo "Creating configs..."

for dataset in "${datasets[@]}"; do
    for i in $(seq 0 $((${#predictors[@]} - 1))); do
        predictor=${predictors[$i]}
        experiment_type=${experiment_types[$i]}
        python $base_file/benchmarks/create_configs.py --predictor $predictor --experiment_type $experiment_type \
            --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
            --dataset $dataset --config_type predictor --search_space $search_space
    done
done

echo "created configs"

# Create a job array file
job_array_file="job_array_darts_${start_seed}.txt"
> $job_array_file

# Populate the job array file
for dataset in "${datasets[@]}"; do
    for t in $(seq $start_seed $end_seed); do
        for i in $(seq 0 $((${#predictors[@]} - 1))); do
            predictor=${predictors[$i]}
            config_file=$out_dir/$dataset/configs/predictors/config\_$predictor\_$t.yaml
            echo "$config_file $predictor $t $dataset" >> $job_array_file
        done
    done
done

# Count total jobs
total_jobs=$(wc -l < $job_array_file)

# Submit the array job
sbatch --array=1-$total_jobs $base_file/benchmarks/scheduler.sh $job_array_file

echo "Array job submitted to scheduler"
