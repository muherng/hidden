#!/bin/bash
import subprocess

# Order: seq_len=$1 chunk_size=$2 n_embd=$3 T1_n_layers=$4 T2_n_layers=$5 n_heads=$6 epochs=$7 lr=$8 vocab_size=$9 num_kv_pairs=$10 power_a= $11 num_train_examples=$12

arg_combinations = [
        ("256", "64", "128", "2", "2", "1", "64", "0.0001", "8192", "8", "1.0", "100000"),
        ("256", "32", "128", "2", "2", "1", "64", "0.0001", "8192", "8", "1.0", "100000"),
        ("128", "64", "128", "2", "2", "1", "64", "0.0001", "8192", "8", "1.0", "100000"),
        ("128", "32", "128", "2", "2", "1", "64", "0.0001", "8192", "8", "1.0", "100000"),
        ("512", "64", "128", "2", "2", "1", "64", "0.0001", "8192", "8", "1.0", "100000"),
        ("512", "32", "128", "2", "2", "1", "64", "0.0001", "8192", "8", "1.0", "100000"),
        ("128", "64", "256", "2", "2", "1", "64", "0.0001", "8192", "8", "1.0", "100000"),
        ("128", "32", "256", "2", "2", "1", "64", "0.0001", "8192", "8", "1.0", "100000"),
        ("512", "64", "256", "2", "2", "1", "64", "0.0001", "8192", "8", "1.0", "100000"),
        ("512", "32", "256", "2", "2", "1", "64", "0.0001", "8192", "8", "1.0", "100000")
    ]
for args in arg_combinations:
    subprocess.run([
        "sbatch",
        f"--job-name=p_{args[0]}_{args[1]}_{args[2]}_kv{args[9]}",
        f"--output=logs/p_{args[0]}_{args[1]}_{args[2]}_kv{args[9]}_a{args[10]}.out",
        f"--error=logs/p_{args[0]}_{args[1]}_{args[2]}_kv{args[9]}_a{args[10]}.err",
        "train_sweepable.sbatch",
        *args,  # unpack the 10 arguments into the command
    ])

# for seq_len in [10, 20, 30]:
#     for chunk_size in [32, 64, 128]:
#         for learning_rate in [0.001, 0.01]:
#             subprocess.run([
#                 "sbatch",
#                 "your_job.sbatch",
#                 str(seq_len),
#                 str(batch_size),
#                 str(learning_rate)
#             ])
