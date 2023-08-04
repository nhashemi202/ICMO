
#python3 ../scripts/train_rl.py --env BabyAI-GoToLocal-v0 --test_env BabyAI-GoToLocal-v0 \
#       --arch NPS_vanilla --seed=1 \
#       --frames=100000000 \
#       --instr_to_rule_mode='linear' --instr-arch='gru' --num_rules=8 --rule_codebook_size=16\
#       --instr-dim=64 --rule_dim=64 --use_all_slots \
#       --use_compositional_split --num_contexts=1 --act_dim=3 --compositional_step=2 \
#       --memory-dim 1024 --procs 32 --flag='NPS vanilla with 1 context, 1204 hidden state size and 2e7 frames on go to local task'

#python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --test_env BabyAI-OpenTwoDoors-v0 \
#       --arch NPS_vanilla --seed=1 \
#       --frames=3000000 \
#       --instr_to_rule_mode='linear' --instr-arch='gru' --num_rules=8 --rule_codebook_size=16\
#       --instr-dim=64 --rule_dim=64 --use_all_slots \
#       --use_compositional_split --num_contexts=1 --act_dim=3 --compositional_step=1 \
#       --memory-dim 1024 --procs 32 --flag='NPS vanilla with 1 context, 1204 hidden state size and 3e7 frames on open two doors'\
#       --model "increase_mem"


#python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --test_env BabyAI-OpenTwoDoors-v0 \
#       --arch NPS_vanilla --seed=1 \
#       --frames=3000000 \
#       --instr_to_rule_mode='linear' --instr-arch='gru' --num_rules=8 --rule_codebook_size=16\
#       --instr-dim=64 --rule_dim=64 --use_all_slots \
#       --use_compositional_split --num_contexts=1 --act_dim=3 --compositional_step=1 \
#       --memory-dim 1024 --procs 32 --flag='NPS vanilla with 1 context, 1204 hidden state size and 3e7 frames on open two doors'\
#       --model "increase_mem_lstm" 

python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --test_env BabyAI-OpenTwoDoors-v0 \
       --arch NPS_vanilla --seed=1 \
       --frames=3000000 \
       --instr_to_rule_mode='linear' --instr-arch='gru' --num_rules=8 --rule_codebook_size=16\
       --instr-dim=64 --rule_dim=64 --use_all_slots \
       --use_compositional_split --num_contexts=1 --act_dim=3 --compositional_step=1 \
       --memory-dim 1024 --procs 32 --flag='NPS vanilla with 1 context, 1204 hidden state size and 3e7 frames on open two doors'\
       --model "128_mem_lstm" --memory-dim 128


