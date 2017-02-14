import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('assets/436-Simon-Doyle-Web-Mobile-Actions.csv')

print df

plt.figure()
df.plot.scatter(x='mobile_actions', y='web_actions');

# print df['action_id'].value_counts()
#
# unique_values = df['action_id'].value_counts()
#
# unique_values.hist(bins=len(unique_values))

# df['action_id'].plot.hist(alpha=0.5, bins=20)

plt.show()


# print df['action_id'].describe()
#
# from numpy import genfromtxt
# my_data = genfromtxt('assets/436-Simon-Doyle-2.csv', delimiter=',')
#
# print my_data

