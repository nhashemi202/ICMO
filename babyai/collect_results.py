
# Python program to explain os.listdir() method 
    
# importing os module 
import os
import torch
import pandas as pd
import csv

# Get the list of all files and directories
# in the root directory
path = "./logs"
dir_list = os.listdir(path)


def write_row(writer, d, L, args, fieldnames, return_mean_test, sr_mean_test, C_A_mean_test, C_W_mean_test):
    row = {'d': d, 'log_len': L}
    for k in fieldnames[2:]:
        if k in vars(args):
            row[k] = vars(args)[k]
        else:
            row[k] = 'N.A.'
    row['return_mean_test'] = return_mean_test
    row['sr_mean_test'] = sr_mean_test
    row['C_A_mean_test'] = C_A_mean_test
    row['C_W_mean_test'] = C_W_mean_test
    writer.writerow(row)

for s in ['GoToLocal', 'OpenTwoDoors']:
    with open(f'summary_BabyAI-{s}-v0.csv', 'w', newline='') as csvfile:
        fieldnames = ['d', 'log_len', 'arch', 'act_dim', 'save_interval','test_env'
            'instr_dim', 'rule_dim', 'no_instr', 'film_d',
            'instr_arch', 'instr_to_rule_mode',  'num_contexts',
            'num_variables', 'seed', 'act_dim', 'use_large_model',
            'use_slot_attention', 'num_rules', 'use_all_slots', 'bottleneck_size', 'use_compositional_split', 
            'flag', 'return_mean_test', 'sr_mean_test', 'C_A_mean_test', 'C_W_mean_test']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for d in dir_list:
            p = f'{path}/{d}'
            if not os.path.exists(f'./models/{d}/args.pkl'): continue
            args = torch.load(f'./models/{d}/args.pkl')
            log_len = len(pd.read_csv(f'./logs/{d}/log.csv'))
            if args.frames >= 900000 and all([k in vars(args) for k in ['env']]) \
                and args.env == f'BabyAI-{s}-v0' and args.flag == 'Test-collapse':

                return_mean_test = []
                sr_mean_test = []
                C_A_mean_test = []
                C_W_mean_test = []
                with open (f'./logs/{d}/log.log', 'rt') as logfile:  
                    for line in logfile:  
                        if '__main__' in line and ': Return ' in line:   
                            print(line.split(': Return ')[1].split(';')[0], d, '============')       
                            return_mean_test.append(float(line.split(': Return ')[1].split(';')[0]))  
                        if '__main__' in line and ': SR ' in line:          
                            sr_mean_test.append(float(line.split(': SR ')[1].split(';')[0]))  
                        if '__main__' in line  and ': C_A ' in line:          
                            C_A_mean_test.append(float(line.split(': C_A ')[1].split(';')[0]))  
                        if '__main__' in line  and ': C_W ' in line:          
                            C_W_mean_test.append(float(line.split(': C_W ')[1].split(';')[0]))   

                write_row(writer, d, log_len, args, fieldnames, return_mean_test, sr_mean_test, C_A_mean_test, C_W_mean_test)

                print('d: ', d, 'test: ', 
                     'arch: ', args.arch, 'instr_dim: ', args.instr_dim, 
                     'rule_dim: ', args.rule_dim if args.no_instr else args.instr_dim, 
                     'instr_arch: ', args.instr_arch, 'no_instr: ', args.no_instr, 
                     'num_context: ', args.num_contexts, 'seed: ', args.seed, 'num_rules: ', args.num_rules,
                     'log_len: ', log_len,
                     #'film_d: ', args.film_d, 
                #     # 'use_large_model: ', args.use_large_model, 
                #     # 'use_slot_attention: ', args.use_slot_attention,
                #     'num_variables: ', args.num_variables
                     )
                # m = torch.load(f'./models/{d}/model.pt')
                # print(m.image_conv.rule_net.instr_to_rule)
                # print('============================================')

