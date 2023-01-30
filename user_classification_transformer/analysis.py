import pandas as pd
import matplotlib.pyplot as plt

# analyze the token lengths in the dataset to decide on the sequence length used by the bert model

token_lengths = pd.read_csv('data/analysis/token_lengths.csv')
token_lengths.columns = ['length', 'freq']

plt.plot(token_lengths['length'], token_lengths['freq'])

plt.xlabel('Token Length')
plt.ylabel('Frequency')

plt.savefig('report/figs/token_lengths.png')

plt.show()

#save plot

