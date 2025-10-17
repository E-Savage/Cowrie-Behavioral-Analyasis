import pandas as pd

df = pd.read_csv('profiles/cowrie_profiles_with_ip_repeat.csv')
print(df.head()) 
print(df.describe())
print(df.info())
print(df.columns)

"""
columns in dataframe that would be useful for clustering:
- ip address
- total session count
- top command
- total commands
- username list
- password list
- command list 
"""

mini_df = df[['src_ip', 'total_sessions', 'top_command', 'commands_count', 'usernames_list', 'passwords_list', 'commands_list']]

df.to_csv('profiles/ip_profiles_mini.csv', index=False)


print(mini_df.head())