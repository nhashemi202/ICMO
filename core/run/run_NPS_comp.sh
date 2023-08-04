# python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --test_env BabyAI-OpenTwoDoors-v0 \
#        --arch NPS_vanilla --num_contexts=1 --seed=42 \
#        --frames=100000000 --use_all_slots \
#        --instr_to_rule_mode='VQ' --instr-arch='VQ' --num_rules=4 --rule_codebook_size=8 \
#        --instr-dim=64 --rule_dim=64 --use_compositional_split

python3 ../scripts/train_rl.py --env BabyAI-GoToLocal-v0 --test_env BabyAI-GoToLocal-v0 \
       --arch NPS_vanilla --num_contexts=1 --seed=42 \
       --frames=100000000 --use_all_slots \
       --instr_to_rule_mode='linear' --instr-arch='gru' --num_rules=8 --rule_codebook_size=16 \
       --instr-dim=64 --rule_dim=64 --use_compositional_split

# python3 ../scripts/train_rl.py --env BabyAI-GoToLocal-v0 \
#        --test_env BabyAI-GoToLocal-v0 --arch expert_filmcnn_1 \
#        --frames=100000000 --instr-dim=64 --seed=42 \
#        --film_d=12 --use_compositional_split

# python3 ../scripts/train_rl.py --env BabyAI-GoToLocal-v0 --test_env BabyAI-GoToLocal-v0 --arch cnn1 \
#        --frames=50000000 --flag='CNN' --no-instr --seed=42 --use_compositional_split
