import os;
import re;
from bs4 import BeautifulSoup;
from nltk.stem import PorterStemmer
import pandas as pd;
import numpy as np;
from scipy.io import loadmat;
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

cwd= os.getcwd();
path = os.path.join(cwd,'data');

def get_sample(fn):
    with open(fn, 'r') as f:
        content = f.read()
    return content

fn = os.path.join(path, 'emailSample1.txt')
content = get_sample(fn);

def word_tokeniize(content):
    '''
    content: str - body of mail
    return: list of tokens (str) e.g. ['>', 'Anyone', 'knows', 'how', 'much', 'it', 'costs', 'to', 'host', 'a']
    '''
    # YOUR_CODE.  Split the content to tokens. You may need re.split()
    # START_CODE
    content = content.replace("\n", " ");
    tokens = content.split(" ");
    # END_CODE

    return tokens

tokens  = word_tokeniize('''> Anyone knows how much it costs to host a web portal ?\n>\nWell, it depends on how many visitors you're expecting.\nThis can be anywhere from less than 10 bucks a month to a couple of $100. \nYou should checkout http://www.rackspace.com/ or perhaps Amazon EC2 \nif youre running something big..\n\nTo unsubscribe yourself from this mailing list, send an email to:\ngroupname-unsubscribe@egroups.com\n\n''')

def lower_case(tokens):
    '''
    tokens: ndarry of str
    return: ndarry of tokens in lower case (str)
    '''
    # YOUR_CODE.  Make all tokens in lower case
    # START_CODE

    tokens = [x.lower() for x in tokens]
    # END_CODE

    return tokens

def normalize_tokens (tokens):
    '''
    tokens: ndarry of str
    return: ndarry of tokens replaced with corresponding unified words
    '''
    # YOUR_CODE.
        # Remove html and other tags
        # mark all numbers "number"
        # mark all  urls as "httpaddr"
        # mark all emails as "emailaddr"
        # replace $ as "dollar"
        # get rid of any punctuation
        # Remove any non alphanumeric characters
    #  You may  need re.sub()
    # START_CODE

    remove_html = lambda x: BeautifulSoup(x, "html.parser").text;
    mark_number = lambda x: re.sub(r'\d+', 'number', x);
    mark_url = lambda x: re.sub(r'http\S+', 'httpaddr', x);
    mark_email = lambda x: re.sub('\S+@\S+', 'emailaddr', x);
    replace_dollar = lambda x: x.replace('$', 'dollar');
    punctuation = lambda x: re.sub('\W+','', x);

    tokens = [remove_html(token) for token in tokens];
    tokens = [mark_number(token) for token in tokens];
    tokens = [mark_url(token) for token in tokens];
    tokens = [mark_email(token) for token in tokens];
    tokens = [replace_dollar(token) for token in tokens];
    tokens = [punctuation(token) for token in tokens];
    # END_CODE

    return tokens

def filter_short_tokens(tokens):
    '''
    tokens: ndarry of str
    return: ndarry of filtered tokens (str)
    '''
    original_tokens_len = len(tokens)

    # YOUR_CODE. Keep only tokens that lenght >0
    # START_CODE
    while '' in tokens:
        tokens.remove('')

    # END_CODE

    print('Original len= {}\nRemaining len= {}'.format(original_tokens_len, len(tokens)))

    return tokens

def stem_tokens(tokens):
    '''
    tokens: ndarry of str
    return: ndarry of stemmed tokens e.g. array(['anyon', 'know', 'how', 'much', 'it', 'cost', 'to', 'host', 'a',
       'web', 'portal', 'well', 'it', 'depend', 'on', 'how', 'mani']...
    '''
    # YOUR_CODE. replace the tokens by stemmed form. You may need PorterStemmer.stem()
    # START_CODE
    ps = PorterStemmer();

    tokens = [ps.stem(token) for token in tokens];
    # END_CODE

    return tokens

def get_vocabulary(fn):
    '''
    fn: str - full path to file
    return: ndarray of str e.g. array(['aa', 'ab', 'abil', ..., 'zdnet', 'zero', 'zip'], dtype=object)
    '''
    vocab_list = pd.read_table(fn, header=None)
    vocab = np.array(vocab_list)[:,1] # first columns is index, select only words column
    print ('len(vocab)= {:,}'.format(len(vocab)))
    return vocab

fn =  os.path.join(path , 'vocab.txt')
vocab = get_vocabulary(fn);

def represent_features(tokens, vocab):
    '''
    tokens: ndarry of str
    tokens: ndarry of str
    return: ndarry of binary values 1 if word from vocabulary is in mail 0 otherwise
    '''
    # YOUR_CODE. Compute the array with 1/0 corresponding to is word from vocabulary in mail
    # START_CODE
    bool_arr = []
    arr_tokens = [];
    for item in range(len(vocab)):
        if vocab[item] in tokens and vocab[item] not in arr_tokens:
            arr_tokens.append(vocab[item]);
            bool_arr.append(1);
        else:
            bool_arr.append(0);

    tokens_represented = bool_arr;
    # END_CODE

    print('{} word(s) from vocab are in the tokens.'.format(np.sum(tokens_represented)))

    return tokens_represented


def preprocess(content, vocab):
    '''
    content: str - body of mail
    vocab: ndarray of str - list of considered words
    '''
    # YOUR_CODE. Compute the array with 1/0 corresponding to is word from vocabulary in mail
    # START_CODE

    # tokenize content
    tokens = word_tokeniize(content);

    # make lower case
    tokens = lower_case(tokens)

    # normalize tokens
    tokens = normalize_tokens(tokens);

    # remove zero words
    tokens = filter_short_tokens(tokens);

    # stem words
    tokens = stem_tokens(tokens)

    # convert to binary array of features
    tokens_represented = represent_features(tokens, vocab);
    # END_CODE

    return tokens_represented;

preprocess (content,vocab)

fn = os.path.join(path , 'spamTrain.mat')

mat = loadmat(fn)
X_train = mat['X']
y_train = mat['y'].ravel()

print('X_train.shape= {}',X_train.shape)
print('y_train.shape= {}',y_train.shape)

fn = os.path.join(path , 'spamTest.mat')
mat = loadmat(fn)
X_test = mat['Xtest']
y_test = mat['ytest'].ravel()

print('X_test.shape= {}',X_test.shape)
print('y_test.shape= {}',y_test.shape)
index = 0
print('Sample with index ={}: \n{}'.format(index, X_train[index]))

C = .1
clf = LinearSVC(C=C)
clf.fit(X_train,y_train)
print ('Score train= {}'.format(clf.score(X_train,y_train)))
print ('Score test= {}'.format(clf.score(X_test,y_test)))

# print('clf.intercept_={}'.format(clf.intercept_))
# print ('clf.coef_={}'.format(clf.coef_))

# YOUR_CODE. Compute top 20 largest coeficients and return the corresponding 20 words from vocabulary
# START_CODE
clf_coef = clf.coef_
top_spam_contributors=[]
clf_coef = sorted(clf_coef[0], reverse=True)
iteration =0;

for item in clf_coef:
    if item in clf.coef_[0]:
        top_spam_contributors.append(vocab[clf.coef_[0].tolist().index(item)]);
        iteration +=1;
    if iteration == 20:
        break;

print("top_spam_contributors: " ,top_spam_contributors);
# END_CODE

for sfn in ['emailSample1.txt', 'emailSample2.txt', 'spamSample1.txt', 'spamSample2.txt']:
    fn = os.path.join(path, sfn)
    content = get_sample(fn)

    # YOUR_CODE.  Preprocess the sample and get prediction 0 or 1 (1 is spam)
    # START_CODE
    predict = preprocess(content, vocab);

    prediction = clf.predict([predict]);
    # END_CODE

    print('{} is {}\n'.format(sfn, ('Not Spam', 'Spam')[prediction[0]]))

print('Latter sample:\n{1}\n{0}\n{1}'.format(content, '=' * 50))