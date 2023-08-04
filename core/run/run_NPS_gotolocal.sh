# python3 ../scripts/train_rl.py --env BabyAI-GoToLocal-v0 --test_env BabyAI-GoToLocal-v0 \
#        --arch NPS_vanilla --num_contexts=1 --seed=42 \
#        --frames=100000000 --use_all_slots \
#        --instr_to_rule_mode='linear' --instr-arch='gru' --num_rules=8 \
#        --instr-dim=64 --rule_dim=64 \
#        --use_slot_attention --num_variables=10\

python3 ../scripts/train_rl.py --env BabyAI-GoToLocal-v0 --test_env BabyAI-GoToLocal-v0 \
       --arch NPS_vanilla --seed=1 \
       --frames=100000000 \
       --instr_to_rule_mode='linear' --instr-arch='gru' --num_rules=8 --rule_codebook_size=16\
       --instr-dim=64 --rule_dim=64 --use_all_slots \
       --use_compositional_split --num_contexts=1 --act_dim=3 --compositional_step=2

# python3 ../scripts/train_rl.py --env BabyAI-GoToLocal-v0 --test_env BabyAI-GoToLocal-v0 \
#        --arch NPS_vanilla --num_contexts=1 --seed=1 \
#        --frames=100000000 --no-instr \
#        --flag='NPS with 1 context and 2e7 frames on babyai, using all slots' \
#        --instr_to_rule_mode='linear' --instr-arch='gru' --num_rules=8 \
#        --instr-dim=64 --rule_dim=64 # single slot
       
# python3 ../scripts/train_rl.py --env BabyAI-GoToLocal-v0 \
#        --test_env BabyAI-GoToLocal-v0 --arch expert_filmcnn_1 \
#        --frames=100000000 --instr-dim=64 --seed=1 \
#        --film_d=12 --use_compositional_split

# python3 ../scripts/train_rl.py --env BabyAI-GoToLocal-v0 --test_env BabyAI-GoToLocal-v0 --arch cnn1 \
#        --frames=50000000 --flag='FIXED Residual instr bug' --no-instr --seed=42
