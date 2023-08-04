##### MHA seed42 PP 64 nc=1 nr=8: BabyAI-GoToLocal_Dynamics_Train-v0_ppo_NPS_vanilla_MHA_mem_seed42_22-07-26-17-17-06
##### FiLM seed1 PP 64 nc=1 nr=8: BabyAI-GoToLocal_Dynamics_Train-v0_ppo_expert_filmcnn_gru_mem_seed1_22-07-28-18-21-23
##### gru seed42 PP 64 nc=1 nr=8: BabyAI-GoToLocal_Dynamics_Train-v0_ppo_NPS_vanilla_gru_mem_seed42_22-07-26-13-40-08
##### VQ seed42 PP 64 nc=1 nr=8: BabyAI-GoToLocal_Dynamics_Train-v0_ppo_NPS_vanilla_VQ_mem_seed42_22-07-26-16-54-30
##### all seed42 PP 64 nc=1 nr=8: BabyAI-GoToLocal_Dynamics_Train-v0_ppo_NPS_vanilla_gru_mem_seed42_22-07-26-13-40-58
##### CNN1 seed42 PP: BabyAI-GoToLocal_Dynamics_Train-v0_ppo_cnn1_gru_mem_seed42_22-07-31-09-54-00
##### FiLM seed42 PP 64 nc=1 nr=8: BabyAI-GoToLocal_Dynamics_Train-v0_ppo_expert_filmcnn_gru_mem_seed42_22-07-31-09-57-28
##### FiLM_d=24 seed42 PP: BabyAI-GoToLocal_Dynamics_Train-v0_ppo_expert_filmcnn_gru_mem_seed42_22-08-03-02-15-13

python3 ../scripts/train_rl.py --continue_pretrained=$1
