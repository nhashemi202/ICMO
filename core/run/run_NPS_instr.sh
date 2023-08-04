# python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --arch NPS_vanilla --num_contexts=1 --instr-dim=64 \
#         --frames=20000000 --use_all_slots --flag='NPS vanilla with 1 context and 2e7 frames on open two doors, using all slots and instructions, second seed'

# python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --arch NPS_vanilla --num_contexts=1 \
#         --instr-dim=64 --rule_dim=64 --instr_to_rule_mode='MHA' --instr-arch='MHA' --num_rules=4 --seed=42 \
#         --frames=9000000 --use_all_slots --flag='NPS vanilla using all slots and instructions MHA, instr-dim=128'

# python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --arch NPS_vanilla --num_contexts=1 \
#         --instr-dim=64 --rule_dim=64 --instr_to_rule_mode='linear' --instr-arch='gru' --num_rules=4 --seed=42\
#         --frames=9000000 --use_all_slots --flag='NPS vanilla using all slots and instructions gru linear, instr-dim=128'


# python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --arch NPS_vanilla --num_contexts=1 \
#         --instr-dim=64 --rule_dim=64 --instr_to_rule_mode='embed_only' --instr-arch='embed_only' --num_rules=9 --seed=42 \
#         --frames=9000000 --use_all_slots --flag='NPS vanilla using all slots and instructions embed_only'

# python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --arch NPS_vanilla --num_contexts=1 --seed=42 \
#         --instr-dim=64 --rule_dim=64 --instr_to_rule_mode='VQ' --instr-arch='VQ' --num_rules=4 --rule_codebook_size=8 \
#         --frames=9000000 --use_all_slots --flag='NPS vanilla using all slots and instructions embed_only'

# python3 ../scripts/train_rl.py --env BabyAI-MoveTwoAcrossS5N2-v0 --arch NPS_vanilla --num_contexts=4 \
#         --instr-dim=128 --rule_dim=128 --instr_to_rule_mode='linear' --instr-arch='gru' --num_rules=8 --seed=42\
#         --frames=20000000 --use_all_slots --flag='NPS vanilla using all slots and instructions gru linear, instr-dim=128'
