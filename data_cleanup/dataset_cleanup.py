import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA

# load dataset
csv_loc = 'data_cleanup/boardgames.csv'
df = pd.read_csv(csv_loc)
print(df.describe())
print("Missing values: %s" % (df.isnull().sum().sum()))

# remove duplicates of ID/name
df.drop_duplicates(subset=['ID', 'Title'], keep="first", inplace=True)
# create new price column, conditional logic for which price to use
price_low = 15
price_high = 150
# -1 if no price data found; whichever value is available if only one price found
df.loc[pd.isna(df['Price']) & pd.isna(df['BGP_Price']), 'Price_Adjusted'] = -1
df.loc[pd.isna(df['Price']) & pd.notna(df['BGP_Price']), 'Price_Adjusted'] = df['BGP_Price']
df.loc[pd.notna(df['Price']) & pd.isna(df['BGP_Price']), 'Price_Adjusted'] = df['Price']
# prefers original dataset price over API scraped price unless one or both values are above/below thresholds
df.loc[pd.isna(df['Price_Adjusted']) & ((df['Price'] <= price_low) | (df['BGP_Price'] <= price_low)), 'Price_Adjusted'] = np.maximum(df['Price'], df['BGP_Price']) #max(df['Price'], df['BGP_Price'])
df.loc[pd.isna(df['Price_Adjusted']) & ((df['Price'] >= price_high) | (df['BGP_Price'] >= price_high)), 'Price_Adjusted'] = np.minimum(df['Price'], df['BGP_Price'])
df.loc[pd.isna(df['Price_Adjusted']) & (((df['Price'] > price_low) & (df['BGP_Price'] > price_low)) & ((df['Price'] < price_high) & (df['BGP_Price'] < price_high))), 'Price_Adjusted'] = df['Price']
# drop values with no price data
df = df[df['Price_Adjusted'] != -1]

# add other engineered features
df['Years_Since_Release'] = 2026 - df['Year']
df.loc[pd.notna(df['Min_Players']) & pd.notna(df['Max_Players']),'Mid_Player'] = (df['Max_Players'] + df['Min_Players']) / 2
df.loc[pd.isna(df['Min_Players']) & pd.notna(df['Max_Players']),'Mid_Player'] = df['Max_Players']
df.loc[pd.notna(df['Min_Players']) & pd.isna(df['Max_Players']),'Mid_Player'] = df['Min_Players']
# create lists of genres in new column 'types' before encoding
df['Type1'] = df['Type1'].fillna("")
df['Type2'] = df['Type2'].fillna("")
df['Type'] = df['Type1'] + "," + df['Type2']
#df.loc[(pd.notna(df['Type1'])) | (pd.notna(df['Type2'])), 'Type'] = df['Type1'] + "," + df['Type2']
# df.loc[(pd.isna(df['Type1'])) & (pd.notna(df['Type2'])), 'Type'] = df['Type2']
# df.loc[(pd.notna(df['Type1'])) & (pd.isna(df['Type2'])), 'Type'] = df['Type1']
df = df.drop(columns=['Type1', 'Type2'])
# one hot encode
mlb = MultiLabelBinarizer()
mlb.fit([["Strategy", "Thematic", "Wargames", "Abstract", "Customizable", "Family", "Children's", "Party"]])
encoded_types = pd.DataFrame(mlb.transform(df['Type'].str.split(',')), columns=mlb.classes_, index=df.index)
df = pd.concat([df, encoded_types], axis=1)
# fill in missing values for min_time and max_time, fix max_time
df['Min_Time'] = df['Min_Time'].fillna(df['Min_Time'].median())
df['Max_Time'] = df['Max_Time'].fillna(df['Max_Time'].median())
df['Max_Time'] = np.maximum(df['Min_Time'], df['Max_Time'])
#df['Max_time'] = max(df['Min_time'], df['Max_time'])

print(df.head)
# remove unnecessary features: id, title, geek_rating, price, BGP_price, num_of_voters, year, min_players, max_players
df = df.drop(columns=['ID','Title','Geek_Rating', 'Price', 'BGP_Price','Num_of_Voters','Year','Min_Players','Max_Players', 'Min_Time', 'Type'])

print(df.describe())
print(df.head)
print("Missing values: %s" % (df.isnull().sum().sum()))
# split into training and test data, fit scaler only on training data, then transform both
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['Price_Adjusted']), df['Price_Adjusted'], test_size=0.2, random_state=42
)
# impute avg_rating, complexity, years_since_release, max_time, min_age; one hot encoding for type(s)
numerical = ['Avg_Rating', 'Complexity', 'Years_Since_Release', 'Mid_Player', 'Max_Time', 'Min_Age']
num_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_trans, numerical)
], remainder='passthrough')
# fit to training data, transform X_test
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
    # cols = df.columns.values.tolist().remove('Price_Adjusted')
    # X_train_df = pd.DataFrame(X_train, columns=cols)
    # print(X_train_df.describe())
    # print(X_train_df.head)
    # print("Missing values: %s" % (X_train_df.isnull().sum().sum()))
# PCA - check which features explain the most variance