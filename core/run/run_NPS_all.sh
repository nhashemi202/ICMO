
# python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --arch NPS_vanilla --no-instr --num_contexts=1 --num_rules=4 --rule_dim=64 \
#        --frames=20000000 --use_all_slots --seed=42 --flag='NPS vanilla with 1 context and 2e7 frames on open two doors, using all slots'

#python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --arch NPS_vanilla --no-instr --num_contexts=1 --act_dim=3 --bottleneck_size=5 \
#        --frames=20000000 --use_all_slots --flag='NPS vanilla with 1 context and 2e7 frames on open two doors, using all slots'

#python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --arch NPS_vanilla --no-instr --num_contexts=1 --act_dim=1 --bottleneck_size=5 \
#        --frames=20000000 --use_all_slots --flag='NPS vanilla with 1 context and 2e7 frames on open two doors, using all slots'

# python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --arch NPS_vanilla --no-instr --num_contexts=1 --act_dim=3 --frames=20000000 --use_all_slots --flag='NPS vanilla with 1 context and 2e7 frames on open two doors, using all slots'


#python3 ../scripts/train_rl.py --env BabyAI-GoToRedBall-v0 --arch NPS_vanilla --no-instr --num_contexts=1 \
#        --frames=20000000 --use_all_slots --flag='NPS vanilla with 1 context and 2e7 using all slots, GoToRedBall' --log-interval=1

# python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --use_slot_rnn --arch NPS_vanilla --no-instr --num_contexts=1 --act_dim=3\
#         --frames=2500000 --use_all_slots --flag='Main NPS, Debug'

# python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --use_slot_rnn --arch NPS_vanilla --no-instr --num_contexts=1 --act_dim=1\
#         --frames=2500000 --use_all_slots --flag='Main NPS, Debug'

# python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --use_slot_rnn --arch NPS_vanilla --no-instr --num_contexts=1 --act_dim=3 --bottleneck_size=20 \
#         --frames=2500000 --use_all_slots --flag='Main NPS, Debug'

# python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --use_slot_rnn --arch NPS_vanilla --no-instr --num_contexts=1 --act_dim=1 --bottleneck_size=20 \
#         --frames=2500000 --use_all_slots --flag='Main NPS, Debug'


#python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --arch NPS_vanilla --no-instr --num_contexts=1 --bottleneck_size=30 --append_coord \
#        --frames=3000000 --use_all_slots --flag='NPS vanilla with 1 context and 2e7 frames on open two doors, using all slots'

# python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --arch NPS_vanilla --no-instr --num_contexts=1 --bottleneck_size=20 --act_dim=3 \
#        --frames=20000000 --use_all_slots --flag='NPS vanilla with 1 context and 2e7 frames on open two doors, using all slots'


# python3 ../scripts/train_rl.py --env BabyAI-MoveTwoAcrossS5N2-v0 --arch NPS_vanilla --no-instr --num_contexts=4 --num_rules=8 --rule_dim=64 \
#        --frames=20000000 --use_all_slots --seed=0 --flag='NPS vanilla with 1 context and 2e7 frames on open two doors, using all slots'
