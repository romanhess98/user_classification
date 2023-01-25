'''
preprocessing
authors provide no further details on this, however, dataset is messy
Problems:
1. duplicates
3. duplicates with different labels
4. some signs were replaced with encodings ("'" --> "‰Ûª")


'''

# import libraries
import pandas as pd
#import os

#os.chdir('C:\\Users\\roman\\PycharmProjects\\semester_3\\TwitterUserClassifier-master\\user_classification_transformer')

# import dataset
df = pd.read_csv('data/df_raw.csv', encoding='latin-1')

# problematic encodings
print("\nSome rows have problematic values. It seems there was something wrong with the encodings there")
print(df.loc[[155, 939, 1330, 166, 2155]])

#rename columns
df.columns = ['description', 'is_gen_pub', 'source']



#remove rows with nan values
df.dropna(inplace=True)

#turn is_gen_pub column into int type
df['is_gen_pub'] = df['is_gen_pub'].astype(int)

#replace all links (starting with "https") in the description columns with "http"
df['description'] = df['description'].str.replace(r'http\S+', 'http', regex=True)

# obtain all rows where 'description' is a duplicate
df_duplicates = df[df.duplicated(['description'], keep=False)]

# find cases where duplicates have inconsistent labels
last_descr = ''
last_label= 0
indices = []

for index, row in df_duplicates.iterrows():
    if row['description'] != last_descr:
        last_descr= row['description']
        last_label = row['is_gen_pub']

    else:
        if row['is_gen_pub'] != last_label:
            indices.append(index)

# filter out rows of dataset_duplicates based on indices
inconsistents = df_duplicates.loc[indices]

uniques = inconsistents['description'].unique()
print(f"\nThere are {len(uniques)} unique descriptions with contradicting labels (sometimes positive, sometimes negative):")

for i in uniques:
    print("    ", i)

"""
with some of these bio texts, it is highly unlikely that they belong to different individuals. 
The authors did not specify how they treated these cases. We will run our analyses on two different datasets:
1. the original dataset
2. the dataset with duplicates removed
    --> we will remove duplicates by keeping the first occurence of a description and the majority label within each 
    source dataset

The dataset was manually labeled. To ensure inter rater reliability, some cases were labeled by multiple raters which 
can lead to inconsistencies.

Our assumption is that the majority label is the correct one.
"""


# split up the dataset into 4 individual ones based on the source
df_boston = df[df['source'] == 'Bostong Marathon Bombing']
df_brussels = df[df['source'] == 'Brussel Airport Bombing']
df_mesa = df[df['source'] == 'Mesa Shooting']
df_quebec = df[df['source'] == 'Quebec Mosque Shooting']
df_random = df[df['source'] == 'Random Samples']


# randomly shuffle each dataset
df_boston = df_boston.sample(frac=1, random_state=42).reset_index(drop=True)
df_brussels = df_brussels.sample(frac=1, random_state=42).reset_index(drop=True)
df_mesa = df_mesa.sample(frac=1, random_state=42).reset_index(drop=True)
df_quebec = df_quebec.sample(frac=1, random_state=42).reset_index(drop=True)
df_random = df_random.sample(frac=1, random_state=42).reset_index(drop=True)

# split each dataset into a train, validation and test set
df_boston_train = df_boston.iloc[:int(0.7*len(df_boston))]
df_boston_val = df_boston.iloc[int(0.7*len(df_boston)):int(0.8*len(df_boston))]
df_boston_test = df_boston.iloc[int(0.8*len(df_boston)):]

df_brussels_train = df_brussels.iloc[:int(0.7*len(df_brussels))]
df_brussels_val = df_brussels.iloc[int(0.7*len(df_brussels)):int(0.8*len(df_brussels))]
df_brussels_test = df_brussels.iloc[int(0.8*len(df_brussels)):]

df_mesa_train = df_mesa.iloc[:int(0.7*len(df_mesa))]
df_mesa_val = df_mesa.iloc[int(0.7*len(df_mesa)):int(0.8*len(df_mesa))]
df_mesa_test = df_mesa.iloc[int(0.8*len(df_mesa)):]

df_quebec_train = df_quebec.iloc[:int(0.7*len(df_quebec))]
df_quebec_val = df_quebec.iloc[int(0.7*len(df_quebec)):int(0.8*len(df_quebec))]
df_quebec_test = df_quebec.iloc[int(0.8*len(df_quebec)):]

df_random_train = df_random.iloc[:int(0.7*len(df_random))]
df_random_val = df_random.iloc[int(0.7*len(df_random)):int(0.8*len(df_random))]
df_random_test = df_random.iloc[int(0.8*len(df_random)):]


# store them as csv files
df_boston.to_csv('data/df_boston.csv', index=False)
df_brussels.to_csv('data/df_brussels.csv', index=False)
df_mesa.to_csv('data/df_mesa.csv', index=False)
df_quebec.to_csv('data/df_quebec.csv', index=False)
df_random.to_csv('data/df_random.csv', index=False)

df_boston_train.to_csv('data/df_boston_train.csv', index=False)
df_boston_val.to_csv('data/df_boston_val.csv', index=False)
df_boston_test.to_csv('data/df_boston_test.csv', index=False)

df_brussels_train.to_csv('data/df_brussels_train.csv', index=False)
df_brussels_val.to_csv('data/df_brussels_val.csv', index=False)
df_brussels_test.to_csv('data/df_brussels_test.csv', index=False)

df_mesa_train.to_csv('data/df_mesa_train.csv', index=False)
df_mesa_val.to_csv('data/df_mesa_val.csv', index=False)
df_mesa_test.to_csv('data/df_mesa_test.csv', index=False)

df_quebec_train.to_csv('data/df_quebec_train.csv', index=False)
df_quebec_val.to_csv('data/df_quebec_val.csv', index=False)
df_quebec_test.to_csv('data/df_quebec_test.csv', index=False)

df_random_train.to_csv('data/df_random_train.csv', index=False)
df_random_val.to_csv('data/df_random_val.csv', index=False)
df_random_test.to_csv('data/df_random_test.csv', index=False)


# find duplicate descriptions
# for each duplicate
# count the number of label=0 and label=1
# remove all occurrences of that description from the dataset
# add a case to the dataset containing the description and the majority label

def replace_duplicates_with_majority_case(dataframe: pd.DataFrame) -> pd.DataFrame:

    # find duplicates in df_boston
    tmp = dataframe[dataframe.duplicated(['description'], keep=False)].copy()

    # store source string
    src = dataframe['source'].iloc[0]

    # remove duplicates from df#
    print(dataframe.shape)
    dataframe.drop_duplicates(subset=['description'], keep=False, inplace=True, ignore_index=True)
    print(dataframe.shape)
    # for each duplicate
    # count the number of label=0 and label=1
    prc = tmp.groupby(by=['description'], as_index=False).mean(numeric_only=True)

    n_inconsistent=0
    for idx, rw in prc.iterrows():
        if rw['is_gen_pub'] != 0 and rw['is_gen_pub'] != 1:
            n_inconsistent += 1
        if rw['is_gen_pub'] >= 0.5:
            majority_case = pd.DataFrame([[rw['description'], 1, src]], columns=['description', 'is_gen_pub', 'source'])

        else:
            majority_case = pd.DataFrame([[rw['description'], 0, src]], columns=['description', 'is_gen_pub', 'source'])

        dataframe = pd.concat([dataframe, majority_case]).copy()

    print(f"There were {str(n_inconsistent)} unique inconsistent descriptions in {src}.")

    return dataframe

# create versions of dfs that have no duplicates and are consistent
df_boston_nd_c = replace_duplicates_with_majority_case(df_boston.copy())
df_brussels_nd_c = replace_duplicates_with_majority_case(df_brussels.copy())
df_mesa_nd_c = replace_duplicates_with_majority_case(df_mesa.copy())
df_quebec_nd_c = replace_duplicates_with_majority_case(df_quebec.copy())
df_random_nd_c = replace_duplicates_with_majority_case(df_random.copy())

# split them into a train and test set
df_boston_nd_c_train = df_boston_nd_c.iloc[:int(0.7*len(df_boston_nd_c))]
df_boston_nd_c_val = df_boston_nd_c.iloc[int(0.7*len(df_boston_nd_c)):int(0.8*len(df_boston_nd_c))]
df_boston_nd_c_test = df_boston_nd_c.iloc[int(0.8*len(df_boston_nd_c)):]

df_brussels_nd_c_train = df_brussels_nd_c.iloc[:int(0.7*len(df_brussels_nd_c))]
df_brussels_nd_c_val = df_brussels_nd_c.iloc[int(0.7*len(df_brussels_nd_c)):int(0.8*len(df_brussels_nd_c))]
df_brussels_nd_c_test = df_brussels_nd_c.iloc[int(0.8*len(df_brussels_nd_c)):]

df_mesa_nd_c_train = df_mesa_nd_c.iloc[:int(0.7*len(df_mesa_nd_c))]
df_mesa_nd_c_val = df_mesa_nd_c.iloc[int(0.7*len(df_mesa_nd_c)):int(0.8*len(df_mesa_nd_c))]
df_mesa_nd_c_test = df_mesa_nd_c.iloc[int(0.8*len(df_mesa_nd_c)):]

df_quebec_nd_c_train = df_quebec_nd_c.iloc[:int(0.7*len(df_quebec_nd_c))]
df_quebec_nd_c_val = df_quebec_nd_c.iloc[int(0.7*len(df_quebec_nd_c)):int(0.8*len(df_quebec_nd_c))]
df_quebec_nd_c_test = df_quebec_nd_c.iloc[int(0.8*len(df_quebec_nd_c)):]

df_random_nd_c_train = df_random_nd_c.iloc[:int(0.7*len(df_random_nd_c))]
df_random_nd_c_val = df_random_nd_c.iloc[int(0.7*len(df_random_nd_c)):int(0.8*len(df_random_nd_c))]
df_random_nd_c_test = df_random_nd_c.iloc[int(0.8*len(df_random_nd_c)):]



# store them as csv files
df_boston_nd_c.to_csv('data/df_boston_nd_c.csv', index=False)
df_brussels_nd_c.to_csv('data/df_brussels_nd_c.csv', index=False)
df_mesa_nd_c.to_csv('data/df_mesa_nd_c.csv', index=False)
df_quebec_nd_c.to_csv('data/df_quebec_nd_c.csv', index=False)
df_random_nd_c.to_csv('data/df_random_nd_c.csv', index=False)

df_boston_nd_c_train.to_csv('data/df_boston_nd_c_train.csv', index=False)
df_boston_nd_c_val.to_csv('data/df_boston_nd_c_val.csv', index=False)
df_boston_nd_c_test.to_csv('data/df_boston_nd_c_test.csv', index=False)

df_brussels_nd_c_train.to_csv('data/df_brussels_nd_c_train.csv', index=False)
df_brussels_nd_c_val.to_csv('data/df_brussels_nd_c_val.csv', index=False)
df_brussels_nd_c_test.to_csv('data/df_brussels_nd_c_test.csv', index=False)

df_mesa_nd_c_train.to_csv('data/df_mesa_nd_c_train.csv', index=False)
df_mesa_nd_c_val.to_csv('data/df_mesa_nd_c_val.csv', index=False)
df_mesa_nd_c_test.to_csv('data/df_mesa_nd_c_test.csv', index=False)

df_quebec_nd_c_train.to_csv('data/df_quebec_nd_c_train.csv', index=False)
df_quebec_nd_c_val.to_csv('data/df_quebec_nd_c_val.csv', index=False)
df_quebec_nd_c_test.to_csv('data/df_quebec_nd_c_test.csv', index=False)

df_random_nd_c_train.to_csv('data/df_random_nd_c_train.csv', index=False)
df_random_nd_c_val.to_csv('data/df_random_nd_c_val.csv', index=False)
df_random_nd_c_test.to_csv('data/df_random_nd_c_test.csv', index=False)

