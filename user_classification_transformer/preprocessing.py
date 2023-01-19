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
df = pd.read_csv('data\\df_raw.csv', encoding='latin-1')

# problematic encodings
print("\nSome rows have problematic values. It seems there was something wrong with the encodings there")
print(df.loc[[155, 939, 1330, 166, 2155]])

#rename columns
df.columns = ['description', 'is_gen_pub', 'source']


#remove rows with nan values
df.dropna(inplace=True)

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

# store them as csv files
df_boston.to_csv('data\\df_boston.csv', index=False)
df_brussels.to_csv('data\\df_brussels.csv', index=False)
df_mesa.to_csv('data\\df_mesa.csv', index=False)
df_quebec.to_csv('data\\df_quebec.csv', index=False)
df_random.to_csv('data\\df_random.csv', index=False)



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
    prc = tmp.groupby(by=['description'], as_index=False).mean()

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

# store them as csv files
df_boston_nd_c.to_csv('data\\df_boston_nd_c.csv', index=False)
df_brussels_nd_c.to_csv('data\\df_brussels_nd_c.csv', index=False)
df_mesa_nd_c.to_csv('data\\df_mesa_nd_c.csv', index=False)
df_quebec_nd_c.to_csv('data\\df_quebec_nd_c.csv', index=False)
df_random_nd_c.to_csv('data\\df_random_nd_c.csv', index=False)
