import pandas as pd
from scipy.stats import entropy
from collections import Counter
import ast 

def safe_list_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    elif isinstance(x, list):
        return x
    else:
        return []

def list_entropy(items):
    """" Computes shannon entropy for a list of items """
    if not items or len(items) == 0:
        return 0
    counts = Counter(items)
    probs = [count / len(items) for count in counts.values()]
    return entropy(probs, base=2)

mini_df = pd.read_csv('profiles/ip_profiles_mini.csv')

# Apply safe_list_eval to convert string representations of lists back to actual lists
mini_df['usernames_list'] = mini_df['usernames_list'].apply(safe_list_eval)
mini_df['passwords_list'] = mini_df['passwords_list'].apply(safe_list_eval)
mini_df['commands_list'] = mini_df['commands_list'].apply(safe_list_eval)

# compute entropy for username, passwords, and command lists
entropy_df = mini_df['src_ip'].copy()

# add columns to mini_df for entropy values to keep them grouped together
mini_df['username_entropy'] = mini_df['usernames_list'].apply(list_entropy)
mini_df['password_entropy'] = mini_df['passwords_list'].apply(list_entropy)
mini_df['command_entropy'] = mini_df['commands_list'].apply(list_entropy)

# add values to make entropy only dataset with ips
entropy_df = mini_df[['src_ip', 'total_sessions', 'username_entropy', 'password_entropy', 'command_entropy']] 


print(entropy_df.head())
print(mini_df.head())

mini_df.to_csv('datasets/ip_profiles_entropy.csv', index=False)
entropy_df.to_csv('datasets/ip_entropies.csv', index=False)
