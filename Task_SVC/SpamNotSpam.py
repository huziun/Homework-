import os;
import re;

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

print(tokens);

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

tokens = lower_case(tokens)
print(tokens);

def normalize_tokens (tokens):
    '''
    tokens: ndarry of str
    return: ndarry of tokens replaced with corresponding unified words
    '''
    # YOUR_CODE.
    regex = re.compile(r'<[^>]+>')
    tokens = regex.sub('', tokens);
        # Remove html and other tags
        # mark all numbers "number"
        # mark all  urls as "httpaddr"
        # mark all emails as "emailaddr"
        # replace $ as "dollar"
        # get rid of any punctuation
        # Remove any non alphanumeric characters
    #  You may  need re.sub()
    # START_CODE
    tokens= None
    # END_CODE

    return tokens

tokens = normalize_tokens(tokens)
print(tokens);