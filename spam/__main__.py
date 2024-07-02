import pandas as pd
from sklearn import model_selection
import re
from collections import defaultdict
import heapq
import functools

def tokenize(text):
    return re.findall(r'[\w+$_\'-]+', text)

def predict(text, probs, new_word=0.4, threshold=0.9):
    tokens = tokenize(text)
    text_probs = {t: probs.get(t, new_word) for t in tokens}
    
    most_interesting = heapq.nlargest(15, tokens, key=lambda token: abs(text_probs.get(token) - 0.5))
    most_interesting_probs = [text_probs.get(t) for t in most_interesting]
    prod = functools.reduce(lambda x, y: x * y, most_interesting_probs)

    prod_complement = functools.reduce(lambda x, y: x * y, map(lambda x: 1 - x, most_interesting_probs))

    result = prod / (prod + prod_complement)
    if result > threshold:
        return 'spam'
    else:
        return 'ham'

if __name__ == "__main__":
    df = pd.read_csv('sms_spam/SMSSpamCollection', sep='\t', names=['class', 'data'])
    # random_state used for deterministic results
    # train, test = model_selection.train_test_split(df, test_size=0.2, random_state=1337)
    train, test = model_selection.train_test_split(df, test_size=0.2)
    # print(f'{train=}')
    # print(f'{test=}')

    # Let's start with the training set and separate out the spam and ham
    # print(train.columns)
    train_ham = train.loc[train['class'] == 'ham']
    train_spam = train.loc[train['class'] == 'spam']

    ham_samplesize = len(train_ham)
    spam_samplesize = len(train_spam)

    print(f"{ham_samplesize=}")
    print(f"{spam_samplesize=}")

    # The above can be used for any methods.
    # We're going to start with Paul Graham's "A Plan for Spam"
    # https://www.paulgraham.com/spam.html

    # Now let's turn the spam and ham into one giant corpus for each
    spam_corpus = train_spam['data'].str.cat(sep=' ').casefold()
    ham_corpus = train_ham['data'].str.cat(sep=' ').casefold()

    # Now we create a hash from token to count for each of the two
    spam_hash = defaultdict(int)
    ham_hash = defaultdict(int)
    # both of those sound like a Denny's order
    spam_tokens = tokenize(spam_corpus)
    ham_tokens = tokenize(ham_corpus)
    all_tokens = list(set(spam_tokens).union(set(ham_tokens)))
    # print(f"spam_tokens length: {len(spam_tokens)}")
    # print(f"ham_tokens length: {len(ham_tokens)}")
    # print(f"all_tokens length: {len(all_tokens)}")

    for token in spam_tokens:
        spam_hash[token] += 1

    for token in ham_tokens:
        ham_hash[token] += 1
    
    # Cool, now the spam probability hash
    spam_probs = {}
    for token in all_tokens:
        g = ham_hash.get(token, 0)
        b = spam_hash.get(token, 0)
        if g + b >= 5:
            s_weight = min(1, b / spam_samplesize)
            h_weight = min(1, g / ham_samplesize)
            spam_probs[token] = max(0.1, min(0.99, s_weight / (s_weight + h_weight)))

    # Alright, now, let's try and predict some stuff
    test['prediction'] = test['data'].apply(lambda text: predict(text, spam_probs))
    test['is_equal'] = test['prediction'] == test['class']


    print(test)
    print(test['is_equal'].mean() * 100)
    print("DONE")
