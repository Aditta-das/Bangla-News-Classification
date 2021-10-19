import pandas as pd
from sklearn import model_selection
import config
from sklearn import preprocessing



def clean_data(name):
    # Replace whitespace between terms with a single space
    processed = name.str.replace(r'\s+', ' ')
    # Remove leading and trailing whitespace
    processed = processed.str.replace(r'^\s+|\s+?$', '')    
    return processed


if __name__ == "__main__":
    # Read csv file
    train = pd.read_csv(config.dataset_path)
    # drop text column, because this dataset already contains clean text
    train.drop("text", axis=1, inplace=True) 
    # more cleaning
    clean_train = clean_data(train["cleanText"])
    '''
    Stopwords refer link
    ref : https://www.ranks.nl/stopwords/bengali
    '''
    with open('bd_stopwords', 'r', encoding='utf-8') as file:
        stopwords = file.readlines()
    stopwords = set(stopwords)
    # stopwords cleaning from sentences
    clean_train = clean_train.apply(lambda x: " ".join(term for term in x.split() if term not in stopwords))
    train['cleanText'] = clean_train
    # label encoding
    l_encod = preprocessing.LabelEncoder()
    train["category"] = l_encod.fit_transform(train["category"])
    y = train.category
    train = train.sample(frac=1).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=train, y=y)):
        train.loc[v_, 'kfold'] = f
    train.to_csv(
            f'{config.path}/{config.csv_name}.csv', 
            index=False
    )
    print("---> Fold Created")