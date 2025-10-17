import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import ast

# load datasets 
main_df = pd.read_csv('datasets/ip_profiles_entropy.csv')
entropy_df = pd.read_csv('datasets/ip_entropies.csv')

main_df['total_sessions'] = pd.to_numeric(main_df['total_sessions'], errors='coerce')
top10 = main_df.sort_values(by='total_sessions', ascending=False).head(10)

# top 10 logins by session counts
plt.bar(top10['src_ip'], top10['total_sessions'])
plt.title("Top 10 session counts by IP")
plt.xlabel("Source IP")
plt.ylabel("Total Sessions")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_sessions/top10_sessions.png')
plt.close()

#  top 10 username entropy by session counts
plt.bar(top10['src_ip'], top10['username_entropy'])
plt.title("Top 10 username entropy by IP")
plt.xlabel("Source IP")
plt.ylabel("Username Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_sessions/top10_username_entropy.png')
plt.close()

#  top 10 password entropy by session counts
plt.bar(top10['src_ip'], top10['password_entropy'])
plt.title("Top 10 password entropy by IP")
plt.xlabel("Source IP")
plt.ylabel("Password Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_sessions/top10_password_entropy.png')
plt.close()

#  top 10 command entropy by session counts
plt.bar(top10['src_ip'], top10['command_entropy'])
plt.title("Top 10 command entropy by IP")
plt.xlabel("Source IP")
plt.ylabel("Command Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_sessions/top10_command_entropy.png')
plt.close()

# top 10 highest uname entropy 
top10_uname_entropy = main_df.sort_values(by='username_entropy', ascending=False).head(10)

# plot top 10 username entropy
plt.bar(top10_uname_entropy['src_ip'], top10_uname_entropy['username_entropy'])
plt.title("Top 10 username entropy by IP")
plt.xlabel("Source IP")
plt.ylabel("Username Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_username_entropy/top10_username_entropy.png')
plt.close()

# plot top 10 uname entropy with session counts
plt.bar(top10_uname_entropy['src_ip'], top10_uname_entropy['total_sessions'])
plt.title("Top 10 username entropy by IP with session counts")
plt.xlabel("Source IP")
plt.ylabel("Total Sessions")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_username_entropy/top10_username_entropy_sessions.png')
plt.close()  

# top 10 uname entropy with password entropy
plt.bar(top10_uname_entropy['src_ip'], top10_uname_entropy['password_entropy'])
plt.title("Top 10 username entropy by IP with password entropy")
plt.xlabel("Source IP")
plt.ylabel("Password Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_username_entropy/top10_username_entropy_password_entropy.png')
plt.close()  

# top 10 uname entropy with command entropy
plt.bar(top10_uname_entropy['src_ip'], top10_uname_entropy['command_entropy'])
plt.title("Top 10 username entropy by IP with command entropy")
plt.xlabel("Source IP")
plt.ylabel("Command Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_username_entropy/top10_username_entropy_command_entropy.png')
plt.close()

# top 10 highest password entropy
top10_pass_entropy = main_df.sort_values(by='password_entropy', ascending=False).head(10)

# plot top 10 password entropy
plt.bar(top10_pass_entropy['src_ip'], top10_pass_entropy['password_entropy'])
plt.title("Top 10 password entropy by IP")
plt.xlabel("Source IP")
plt.ylabel("Password Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_password_entropy/top10_password_entropy.png')
plt.close()  

# plot top 10 password entropy with session counts
plt.bar(top10_pass_entropy['src_ip'], top10_pass_entropy['total_sessions'])
plt.title("Top 10 password entropy by IP with session counts")
plt.xlabel("Source IP")
plt.ylabel("Total Sessions")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_password_entropy/top10_password_entropy_sessions.png')
plt.close()  

# top 10 password entropy with username entropy
plt.bar(top10_pass_entropy['src_ip'], top10_pass_entropy['username_entropy'])
plt.title("Top 10 password entropy by IP with username entropy")
plt.xlabel("Source IP")
plt.ylabel("Username Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_password_entropy/top10_password_entropy_username_entropy.png')
plt.close()

# top 10 password entropy with command entropy
plt.bar(top10_pass_entropy['src_ip'], top10_pass_entropy['command_entropy'])        
plt.title("Top 10 password entropy by IP with command entropy")
plt.xlabel("Source IP")
plt.ylabel("Command Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_password_entropy/top10_password_entropy_command_entropy.png')
plt.close()

# top 10 highest command entropy
top10_command_entropy = main_df.sort_values(by='command_entropy', ascending=False).head(10)

# plot top 10 command entropy
plt.bar(top10_command_entropy['src_ip'], top10_command_entropy['command_entropy'])
plt.title("Top 10 command entropy by IP")
plt.xlabel("Source IP")
plt.ylabel("Command Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_command_entropy/top10_command_entropy.png')
plt.close()  

# plot top 10 command entropy with session counts
plt.bar(top10_command_entropy['src_ip'], top10_command_entropy['total_sessions'])
plt.title("Top 10 command entropy by IP with session counts")
plt.xlabel("Source IP")
plt.ylabel("Total Sessions")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_command_entropy/top10_command_entropy_sessions.png')
plt.close()

# top 10 command entropy with username entropy
plt.bar(top10_command_entropy['src_ip'], top10_command_entropy['username_entropy'])
plt.title("Top 10 command entropy by IP with username entropy")
plt.xlabel("Source IP")
plt.ylabel("Username Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_command_entropy/top10_command_entropy_username_entropy.png')
plt.close()

# top 10 command entropy with password entropy
plt.bar(top10_command_entropy['src_ip'], top10_command_entropy['password_entropy'])
plt.title("Top 10 command entropy by IP with password entropy")
plt.xlabel("Source IP")
plt.ylabel("Password Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/most_command_entropy/top10_command_entropy_password_entropy.png')
plt.close()




#########################################################################################
# now we gonna to 10 least for each category

# least 10 sorted by session counts
least10 = main_df.sort_values(by='total_sessions', ascending=True).head(10)

# plot least 10 session counts
plt.bar(least10['src_ip'], least10['total_sessions'])
plt.title("Least 10 session counts by IP")
plt.xlabel("Source IP")
plt.ylabel("Total Sessions")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_sessions/least10_sessions.png')
plt.close()

# least 10 username entropy by session counts
plt.bar(least10['src_ip'], least10['username_entropy'])
plt.title("Least 10 username entropy by IP")
plt.xlabel("Source IP")
plt.ylabel("Username Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_sessions/least10_username_entropy.png')
plt.close() 

# least 10 password entropy by session counts
plt.bar(least10['src_ip'], least10['password_entropy'])
plt.title("Least 10 password entropy by IP")
plt.xlabel("Source IP")
plt.ylabel("Password Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_sessions/least10_password_entropy.png')
plt.close()

# least 10 command entropy by session counts
plt.bar(least10['src_ip'], least10['command_entropy'])
plt.title("Least 10 command entropy by IP")
plt.xlabel("Source IP")
plt.ylabel("Command Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_sessions/least10_command_entropy.png')
plt.close() 

# least 10 highest uname entropy 
least10_uname_entropy = main_df.sort_values(by='username_entropy', ascending=True).head(10)

# plot least 10 username entropy
plt.bar(least10_uname_entropy['src_ip'], least10_uname_entropy['username_entropy'])
plt.title("Least 10 username entropy by IP")
plt.xlabel("Source IP")
plt.ylabel("Username Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_username_entropy/least10_username_entropy.png')
plt.close() 

# plot least 10 uname entropy with session counts
plt.bar(least10_uname_entropy['src_ip'], least10_uname_entropy['total_sessions'])
plt.title("Least 10 username entropy by IP with session counts")
plt.xlabel("Source IP")
plt.ylabel("Total Sessions")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_username_entropy/least10_username_entropy_sessions.png')
plt.close() 

# top 10 uname entropy with password entropy
plt.bar(least10_uname_entropy['src_ip'], least10_uname_entropy['password_entropy'])
plt.title("Least 10 username entropy by IP with password entropy")
plt.xlabel("Source IP")
plt.ylabel("Password Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_username_entropy/least10_username_entropy_password_entropy.png')
plt.close()

# top 10 uname entropy with command entropy
plt.bar(least10_uname_entropy['src_ip'], least10_uname_entropy['command_entropy'])
plt.title("Least 10 username entropy by IP with command entropy")
plt.xlabel("Source IP")
plt.ylabel("Command Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()  
plt.savefig('plots/october_17_2025/least_username_entropy/least10_username_entropy_command_entropy.png')
plt.close()

# least 10 highest password entropy
least10_pass_entropy = main_df.sort_values(by='password_entropy', ascending=True).head(10)  

# plot least 10 password entropy
plt.bar(least10_pass_entropy['src_ip'], least10_pass_entropy['password_entropy'])
plt.title("Least 10 password entropy by IP")
plt.xlabel("Source IP")
plt.ylabel("Password Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_password_entropy/least10_password_entropy.png')
plt.close()

# plot least 10 password entropy with session counts
plt.bar(least10_pass_entropy['src_ip'], least10_pass_entropy['total_sessions'])
plt.title("Least 10 password entropy by IP with session counts")
plt.xlabel("Source IP")
plt.ylabel("Total Sessions")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_password_entropy/least10_password_entropy_sessions.png')
plt.close()

# top 10 password entropy with username entropy
plt.bar(least10_pass_entropy['src_ip'], least10_pass_entropy['username_entropy'])
plt.title("Least 10 password entropy by IP with username entropy")
plt.xlabel("Source IP")     
plt.ylabel("Username Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_password_entropy/least10_password_entropy_username_entropy.png')
plt.close()

# top 10 password entropy with command entropy
plt.bar(least10_pass_entropy['src_ip'], least10_pass_entropy['command_entropy'])
plt.title("Least 10 password entropy by IP with command entropy")
plt.xlabel("Source IP")
plt.ylabel("Command Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_password_entropy/least10_password_entropy_command_entropy.png')
plt.close() 

# least 10 highest command entropy
least10_command_entropy = main_df.sort_values(by='command_entropy', ascending=True).head(10)

# plot least 10 command entropy
plt.bar(least10_command_entropy['src_ip'], least10_command_entropy['command_entropy'])
plt.title("Least 10 command entropy by IP")
plt.xlabel("Source IP")
plt.ylabel("Command Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_command_entropy/least10_command_entropy.png')
plt.close() 

# plot least 10 command entropy with session counts
plt.bar(least10_command_entropy['src_ip'], least10_command_entropy['total_sessions'])
plt.title("Least 10 command entropy by IP with session counts")
plt.xlabel("Source IP")
plt.ylabel("Total Sessions")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_command_entropy/least10_command_entropy_sessions.png')
plt.close() 

# top 10 command entropy with username entropy
plt.bar(least10_command_entropy['src_ip'], least10_command_entropy['username_entropy'])
plt.title("Least 10 command entropy by IP with username entropy")
plt.xlabel("Source IP")
plt.ylabel("Username Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_command_entropy/least10_command_entropy_username_entropy.png')
plt.close()

# top 10 command entropy with password entropy
plt.bar(least10_command_entropy['src_ip'], least10_command_entropy['password_entropy'])
plt.title("Least 10 command entropy by IP with password entropy")
plt.xlabel("Source IP")
plt.ylabel("Password Entropy")
plt.xticks(rotation=90)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('plots/october_17_2025/least_command_entropy/least10_command_entropy_password_entropy.png')
plt.close()



