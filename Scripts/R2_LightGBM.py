import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix, vstack
import lightgbm as lgb
import gc
import re
from multiprocessing import Pool
import multiprocessing as mp
import psutil



class r_data(object):

    def __init__(self, MAX_LEN, vec='hash'):
        self.arrGood = [
            'absolutely',
            'abundant',
            'accept',
            'acclaimed',
            'accomplishment',
            'achievement',
            'action',
            'active',
            'activist',
            'acumen',
            'adjust',
            'admire',
            'adopt',
            'adorable',
            'adored',
            'adventure',
            'affirmation',
            'affirmative',
            'affluent',
            'agree',
            'airy',
            'alive',
            'alliance',
            'ally',
            'alter',
            'amaze',
            'amity',
            'animated',
            'answer',
            'appreciation',
            'approve',
            'aptitude',
            'artistic',
            'assertive',
            'astonish',
            'astounding',
            'astute',
            'attractive',
            'authentic',
            'basic',
            'beaming',
            'beautiful',
            'believe',
            'benefactor',
            'benefit',
            'bighearted',
            'blessed',
            'bliss',
            'bloom',
            'bountiful',
            'bounty',
            'brave',
            'bright',
            'brilliant',
            'bubbly',
            'bunch',
            'burgeon',
            'calm',
            'care',
            'celebrate',
            'certain',
            'change',
            'character',
            'charitable',
            'charming',
            'cheer',
            'cherish',
            'clarity',
            'classy',
            'clean',
            'clever',
            'closeness',
            'commend',
            'companionship',
            'complete',
            'comradeship',
            'confident',
            'connect',
            'connected',
            'constant',
            'content',
            'conviction',
            'copious',
            'core',
            'coupled',
            'courageous',
            'creative',
            'cuddle',
            'cultivate',
            'cure',
            'curious',
            'cute',
            'dazzling',
            'delight',
            'direct',
            'discover',
            'distinguished',
            'divine',
            'donate',
            'each day',
            'eager',
            'earnest',
            'easy',
            'ecstasy',
            'effervescent',
            'efficient',
            'effortless',
            'electrifying',
            'elegance',
            'embrace',
            'encompassing',
            'encourage',
            'endorse',
            'energized',
            'energy',
            'enjoy',
            'enormous',
            'enthuse',
            'enthusiastic',
            'entirely',
            'essence',
            'established',
            'esteem',
            'everyday',
            'everyone',
            'excited',
            'exciting',
            'exhilarating',
            'expand',
            'explore',
            'express',
            'exquisite',
            'exultant',
            'faith',
            'familiar',
            'family',
            'famous',
            'feat',
            'fit',
            'flourish',
            'fortunate',
            'fortune',
            'freedom',
            'fresh',
            'friendship',
            'full',
            'funny',
            'gather',
            'generous',
            'genius',
            'genuine',
            'give',
            'glad',
            'glow',
            'good',
            'gorgeous',
            'grace',
            'graceful',
            'gratitude',
            'green',
            'grin',
            'group',
            'grow',
            'handsome',
            'happy',
            'harmony',
            'healed',
            'healing',
            'healthful',
            'healthy',
            'heart',
            'hearty',
            'heavenly',
            'helpful',
            'here',
            'hold',
            'holy',
            'honest',
            'honor',
            'hug',
            'idea',
            'ideal',
            'imaginative',
            'increase',
            'incredible',
            'independent',
            'ingenious',
            'innate',
            'innovate',
            'inspire',
            'instantaneous',
            'instinct',
            'intellectual',
            'intelligence',
            'intuitive',
            'inventive',
            'joined',
            'jovial',
            'joy',
            'jubilation',
            'keen',
            'key',
            'kind',
            'kiss',
            'knowledge',
            'laugh',
            'leader',
            'learn',
            'legendary',
            'light',
            'lively',
            'love',
            'loveliness',
            'lucidity',
            'lucrative',
            'luminous',
            'maintain',
            'marvelous',
            'master',
            'meaningful',
            'meditate',
            'mend',
            'metamorphosis',
            'mind-blowing',
            'miracle',
            'mission',
            'modify',
            'motivate',
            'moving',
            'natural',
            'nature',
            'nourish',
            'nourished',
            'novel',
            'now',
            'nurture',
            'nutritious',
            'one',
            'open',
            'openhanded',
            'optimistic',
            'paradise',
            'party',
            'peace',
            'perfect',
            'phenomenon',
            'pleasure',
            'plenteous',
            'plentiful',
            'plenty',
            'plethora',
            'poise',
            'polish',
            'popular',
            'positive',
            'powerful',
            'prepared',
            'pretty',
            'principle',
            'productive',
            'project',
            'prominent',
            'prosperous',
            'protect',
            'proud',
            'purpose',
            'quest',
            'quick',
            'quiet',
            'ready',
            'recognize',
            'refinement',
            'refresh',
            'rejoice',
            'rejuvenate',
            'relax',
            'reliance',
            'rely',
            'remarkable',
            'renew',
            'renowned',
            'replenish',
            'resolution',
            'resound',
            'resources',
            'respect',
            'restore',
            'revere',
            'revolutionize',
            'rewarding',
            'rich',
            'robust',
            'rousing',
            'safe',
            'secure',
            'see',
            'sensation',
            'serenity',
            'shift',
            'shine',
            'show',
            'silence',
            'simple',
            'sincerity',
            'smart',
            'smile',
            'smooth',
            'solution',
            'soul',
            'sparkling',
            'spirit',
            'spirited',
            'spiritual',
            'splendid',
            'spontaneous',
            'still',
            'stir',
            'strong',
            'style',
            'success',
            'sunny',
            'support',
            'sure',
            'surprise',
            'sustain',
            'synchronized',
            'team',
            'thankful',
            'therapeutic',
            'thorough',
            'thrilled',
            'thrive',
            'today',
            'together',
            'tranquil',
            'transform',
            'triumph',
            'trust',
            'truth',
            'unity',
            'unusual',
            'unwavering',
            'upbeat',
            'value',
            'vary',
            'venerate',
            'venture',
            'very',
            'vibrant',
            'victory',
            'vigorous',
            'vision',
            'visualize',
            'vital',
            'vivacious',
            'voyage',
            'wealthy',
            'welcome',
            'well',
            'whole',
            'wholesome',
            'willing',
            'wonder',
            'wonderful',
            'wondrous',
            'xanadu',
            'yes',
            'yippee',
            'young',
            'youth',
            'youthful',
            'zeal',
            'zest',
            'zing',
            'zip'
        ]


        self.arrBad = [
            'acrotomophilia',
            'anal',
            'anilingus',
            'anus',
            'arsehole',
            'ass',
            'asshole',
            'assmunch',
            'autoerotic',
            'babeland',
            'bangbros',
            'bareback',
            'barenaked',
            'bastardo',
            'bastinado',
            'bbw',
            'bdsm',
            'bestiality',
            'bimbos',
            'birdlock',
            'bitch',
            'blumpkin',
            'bollocks',
            'bondage',
            'boner',
            'boob',
            'boobs',
            'bukkake',
            'bulldyke',
            'bullet vibe',
            'bung hole',
            'bunghole',
            'busty',
            'butt',
            'buttcheeks',
            'butthole',
            'camgirl',
            'camslut',
            'camwhore',
            'carpetmuncher',
            'circlejerk',
            'clit',
            'clitoris',
            'clusterfuck',
            'cock',
            'cocks',
            'coprolagnia',
            'coprophilia',
            'cornhole',
            'cum',
            'cumming',
            'cunnilingus',
            'cunt',
            'darkie',
            'daterape',
            'deepthroat',
            'dick',
            'dildo',
            'doggiestyle',
            'doggystyle',
            'dolcett',
            'domination',
            'dominatrix',
            'dommes',
            'ecchi',
            'ejaculation',
            'erotic',
            'erotism',
            'escort',
            'eunuch',
            'faggot',
            'fecal',
            'felch',
            'fellatio',
            'feltch',
            'femdom',
            'figging',
            'fingering',
            'fisting',
            'footjob',
            'frotting',
            'fuck',
            'fucking',
            'fudgepacker',
            'futanari',
            'gay',
            'genitals',
            'goatcx',
            'goatse',
            'gokkun',
            'goodpoop',
            'goregasm',
            'grope',
            'guro',
            'handjob',
            'hardcore',
            'hentai',
            'homoerotic',
            'honkey',
            'hooker',
            'kill',
            'murder',
            'fat',
            'humping',
            'incest',
            'intercourse',
            'jack',
            'jerk',
            'jigaboo',
            'jiggaboo',
            'jiggerboo',
            'jizz',
            'juggs',
            'kike',
            'kinbaku',
            'kinkster',
            'kinky',
            'knobbing',
            'lolita',
            'lovemaking',
            'masturbate',
            'motherfucker',
            'muffdiving',
            'nambla',
            'nawashi',
            'negro',
            'neonazi',
            'nigga',
            'nigger',
            'nimphomania',
            'nipple',
            'nipples',
            'nude',
            'nudity',
            'nympho',
            'nymphomania',
            'octopussy',
            'omorashi',
            'orgasm',
            'orgy',
            'paedophile',
            'panties',
            'panty',
            'pedobear',
            'pedophile',
            'pegging',
            'penis',
            'pissing',
            'pisspig',
            'playboy',
            'ponyplay',
            'poof',
            'poopchute',
            'porn',
            'porno',
            'pornography',
            'pthc',
            'pubes',
            'pussy',
            'queaf',
            'raghead',
            'rape',
            'raping',
            'rapist',
            'rectum',
            'cowgirl',
            'rimjob',
            'rimming',
            'sadism',
            'scat',
            'schlong',
            'scissoring',
            'semen',
            'sex',
            'sexo',
            'sexy',
            'beaver',
            'pussy',
            'shemale',
            'shibari',
            'shit',
            'shota',
            'shrimping',
            'slanteye',
            'slut',
            'smut',
            'snatch',
            'snowballing',
            'sodomize',
            'sodomy',
            'spic',
            'spooge',
            'strapon',
            'strappado',
            'strip',
            'suck',
            'sucks',
            'suicide',
            'sultry',
            'swastika',
            'swinger',
            'threesome',
            'throating',
            'tit',
            'tits',
            'titties',
            'titty',
            'topless',
            'tosser',
            'towelhead',
            'tranny',
            'tribadism',
            'tubgirl',
            'tushy',
            'twat',
            'twink',
            'twinkie',
            'undressing',
            'upskirt',
            'urophilia',
            'vagina',
            'vibrator',
            'vorarephilia',
            'voyeur',
            'vulva',
            'wank',
            'wetback',
            'xx',
            'xxx',
            'yaoi',
            'yiffy',
        ]

        self.CHARS_TO_REMOVE = '!¡"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
        self.MAX_LEN = MAX_LEN
        self.STOP_WORDS = list(stopwords.words('english'))
        self.num_partitions = psutil.cpu_count()
        self.num_cores = psutil.cpu_count()
        
        if vec == 'hash':
            self.vectorizer = HashingVectorizer(n_features=self.MAX_LEN,
                                                analyzer='word',
                                                lowercase=False)
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=(1,2),
                                              min_df=3, max_df=0.9,
                                              strip_accents='unicode',
                                              use_idf=1,
                                              smooth_idf=1,
                                              sublinear_tf=1,
                                              max_features=self.MAX_LEN,
                                              lowercase=False)

    def read_data(self, path, tr=True, n_rows=None):

        if tr:
            return pd.read_csv(path, usecols=['comment_text', 'target']).fillna(' ')
        else:
            return pd.read_csv(path, usecols=['comment_text']).fillna(' ')
        
    def df_parallelize_run(self, df, func, vector=False):
    
        df_split = np.array_split(df, self.num_partitions)
        pool = Pool(self.num_cores)
        if vector:
            df = vstack(pool.map(func, df_split), format='csr')
        else:
            df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df


    def clean_text(self, df):
        
        def remove_stop_words(text, sw):
            text = ' '.join([word for word in text.split() if word not in sw])
            return text
        
        def remove_noise_chars(text, chars):
            text = ''.join([word for word in text if word not in chars])
            return text
        
        def c_t2(text):
            text = text.lower()
            text = re.sub(r"what's", "what is ", text)
            text = re.sub(r"\'s", " ", text)
            text = re.sub(r"\'ve", " have ", text)
            text = re.sub(r"can't", "cannot ", text)
            text = re.sub(r"n't", " not ", text)
            text = re.sub(r"i'm", "i am ", text)
            text = re.sub(r"\'re", " are ", text)
            text = re.sub(r"\'d", " would ", text)
            text = re.sub(r"\'ll", " will ", text)
            text = re.sub(r"\'scuse", " excuse ", text)
            text = re.sub('\W', ' ', text)
            text = re.sub('\s+', ' ', text)
            text = text.strip(' ')
            return text
        
        df["ast"] = df["comment_text"].apply(lambda x: x.count('*'))
        df["ex"] = df["comment_text"].apply(lambda x: x.count('!'))
        df["qu"] = df["comment_text"].apply(lambda x: x.count('?'))
        df["ar"] = df["comment_text"].apply(lambda x: x.count('@'))
        df["ha"] = df["comment_text"].apply(lambda x: x.count('#'))
        df["len_pr"] = df["comment_text"].apply(lambda x: len(x))
        df["num_words"] = df["comment_text"].apply(lambda x: len(x.split()))
        df["num_upper"] = df["comment_text"].apply(lambda x: len([i for i in x.split() if i.isupper()]))
        df["num_lower"] = df["comment_text"].apply(lambda x: len([i for i in x.split() if i.islower()]))
        df["len_max_word"] = df["comment_text"].apply(lambda x: max([len(i) for i in x.split()]))
        df["len_min_word"] = df["comment_text"].apply(lambda x: min([len(i) for i in x.split()]))
        df['num_unique_words'] = df['comment_text'].apply(lambda x: len(set(w for w in x.split())))
        df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
        df['num_smilies'] = df['comment_text'].apply(lambda x: sum(x.count(w) for w in [':-)', ':)', ';-)', ';)']))
        # Add count uppercase, lowercase, etc
        
        df["comment_text"] = df["comment_text"].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
        df["comment_text"] = df["comment_text"].apply(lambda x: c_t2(x))
        df["comment_text"] = df["comment_text"].apply(lambda x: remove_stop_words(x, self.STOP_WORDS))
        df["comment_text"] = df["comment_text"].apply(lambda x: remove_noise_chars(x, self.CHARS_TO_REMOVE))
        
        df["num_bad_words"] = df["comment_text"].apply(lambda x: len([word for word in x.split() if word in self.arrBad]))
        df["num_good_words"] = df["comment_text"].apply(lambda x: len([word for word in x.split() if word in self.arrGood]))
        
        df['bad_ratio'] = df.num_bad_words / df.num_words
        df['good_ratio'] = df.num_good_words / df.num_words
        df['bad_p_good'] = df.num_bad_words + df.num_good_words
        df['bad_m_good'] = df.num_bad_words - df.num_good_words
        df['ratio_max_len'] = df.len_max_word / df.len_pr

        return df

    def vector(self, text_data, train=True, to_dataframe=False):

        if train:
            self.vectorizer.fit(text_data)
            
        X_words = self.vectorizer.transform(text_data)
        
        if to_dataframe:
            X_words = pd.DataFrame(X_words)            

        return X_words

    def final_X(self, X_words, data_extra):

#        X = pd.concat([X_words, data_extra], axis=1)
        X = np.concatenate((X_words, data_extra), axis = 1)

        return X


class r_lgb_model(object):

    def __init__(self, k, params):

        self.k = k
        self.skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)

        self.params = params
        self.lgb_model = lgb.LGBMClassifier(**self.params)

        self. model_list = [0] * self.k

    def _fit(self, X, y, verbose, esr):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.25,
                                                            random_state=42)
        
        self.lgb_model.fit(X_train,
                          y_train,
                          eval_set=[(X_test, y_test)],
                          verbose=verbose,
                          early_stopping_rounds=esr)
        
        self.ft_importances = self.lgb_model.feature_importances_



if __name__ == '__main__':

    MAX_LEN = 30000
    k = 3
    params = {
        'max_depth': -1,
        'metric': 'auc',
        'n_estimators': 2000,
        'learning_rate': 0.1,
#        'num_leaves': 75,
        'colsample_bytree': 0.4,
        'objective': 'binary',
        'n_jobs': -1,
        'seed': 42,
        'bagging_fraction': 0.7,
        'lambda_l1': 0,
        'lambda_l2': 0,
    }
    verbose = 50
    EARLY_STOPPING_ROUNDS = 50
    FT_SEL = False
    
    print('\n### TFIDF ###')          
    print('\n| Data processing...\n')
    data = r_data(MAX_LEN, vec='tfidf')
    
    print('\t- Reading data...')
    train = data.read_data('../data/train.csv')\
#    .sample(100000, random_state=42).reset_index(drop=True)
    test = data.read_data('../data/test.csv', tr=False)
    
    print('\t- Cleaning data...')
#    train = data.clean_text(train)
#    test = data.clean_text(test)
    train = data.df_parallelize_run(train, data.clean_text)
    test = data.df_parallelize_run(test, data.clean_text)
    
    print('\t- Vectorizer...')
    data.vector(pd.concat([train['comment_text'], test['comment_text']]))
    X_words = data.vector(train['comment_text'].values, train=False)
    X_words_test = data.vector(test['comment_text'].values, train=False)
    
    print('\t- Generating final datasets...')
    X_cols = ['ast', 'ex', 'qu', 'ar', 'ha', 'len_pr',
           'num_words', 'len_max_word', 'len_min_word', 'num_bad_words',
           'num_good_words', 'bad_ratio', 'good_ratio', 'bad_p_good', 'bad_m_good',
           'ratio_max_len']
    
    y_train = np.where(train['target'] >= 0.5, 1, 0)
    extra_data = csr_matrix(train[X_cols])
    del train
    gc.collect()

    X_train = hstack([X_words, extra_data]).tocsr()
    
    del X_words
    del extra_data
    gc.collect()      
    
    print('\n\n| Modeling...\n')
    model = r_lgb_model(k, params)
    print('\t- Fitting...')
    model._fit(X_train, y_train, verbose, EARLY_STOPPING_ROUNDS)
    
    if FT_SEL:
    
        df_imp = pd.DataFrame({'feature': [i for i in range(X_train.shape[1])],
                               'importance': model.lgb_model.feature_importances_})\
                .sort_values('importance', ascending = False)
                
        X_train = X_train[:, df_imp.feature[:1000].to_list()]
        
        print('\n\n| Modeling with FT Selection...\n')
        model = r_lgb_model(k, params)
        print('\t- Fitting...')
        model._fit(X_train, y_train, verbose, EARLY_STOPPING_ROUNDS)
    
    del X_train
    del y_train
    gc.collect()
    
    
    print('\n| Predictions...\n')    
    print('\t- Generating final datasets...')
    extra_data = csr_matrix(test[X_cols])
    del test
    gc.collect()

    X_test = hstack([X_words_test, extra_data]).tocsr()
    
    del X_words_test
    del extra_data
    gc.collect()
    
    if FT_SEL:
        X_test = X_test[:, df_imp.feature[:1000].to_list()]
        
    print('\t- Making predictions...')
    preds_tdidf = model.lgb_model.predict_proba(X_test)[:,1]
    
    del X_test
    gc.collect()
    
    
    
    print('\n### HASHING ###')          
    print('\n| Data processing...\n')
    data = r_data(MAX_LEN, vec='hash')
    
    print('\t- Reading data...')
    train = data.read_data('../data/train.csv')\
#    .sample(100000, random_state=42).reset_index(drop=True)
    test = data.read_data('../data/test.csv', tr=False)
    
    print('\t- Cleaning data...')
#    train = data.clean_text(train)
#    test = data.clean_text(test)
    train = data.df_parallelize_run(train, data.clean_text)
    test = data.df_parallelize_run(test, data.clean_text)
    
    print('\t- Vectorizer...')
    data.vector(pd.concat([train['comment_text'], test['comment_text']]))
        
    X_words = data.vector(train['comment_text'].values, train=False)
    X_words_test = data.vector(test['comment_text'].values, train=False)
    
    print('\t- Generating final datasets...')
    X_cols = ['ast', 'ex', 'qu', 'ar', 'ha', 'len_pr',
           'num_words', 'len_max_word', 'len_min_word', 'num_bad_words',
           'num_good_words', 'bad_ratio', 'good_ratio', 'bad_p_good', 'bad_m_good',
           'ratio_max_len']
    
    y_train = np.where(train['target'] >= 0.5, 1, 0)
    extra_data = csr_matrix(train[X_cols])
    del train
    gc.collect()

    X_train = hstack([X_words, extra_data]).tocsr()
    
    del X_words
    del extra_data
    gc.collect()      
    
    print('\n\n| Modeling...\n')
    model = r_lgb_model(k, params)
    print('\t- Fitting...')
    model._fit(X_train, y_train, verbose, EARLY_STOPPING_ROUNDS)
    
    if FT_SEL:
    
        df_imp = pd.DataFrame({'feature': [i for i in range(X_train.shape[1])],
                               'importance': model.lgb_model.feature_importances_})\
                .sort_values('importance', ascending = False)
                
        X_train = X_train[:, df_imp.feature[:1000].to_list()]
        
        print('\n\n| Modeling with FT Selection...\n')
        model = r_lgb_model(k, params)
        print('\t- Fitting...')
        model._fit(X_train, y_train, verbose, EARLY_STOPPING_ROUNDS)
    
    del X_train
    del y_train
    gc.collect()
    
    
    print('\n| Predictions...\n')    
    print('\t- Generating final datasets...')
    extra_data = csr_matrix(test[X_cols])
    del test
    gc.collect()

    X_test = hstack([X_words_test, extra_data]).tocsr()
    
    del X_words_test
    del extra_data
    gc.collect()
    
    if FT_SEL:
        X_test = X_test[:, df_imp.feature[:1000].to_list()]
        
    print('\t- Making predictions...')
    preds_hash = model.lgb_model.predict_proba(X_test)[:,1]
    
    del X_test
    gc.collect()
    
    
    
    preds = (preds_tdidf + preds_hash) / 2
    
    print('\n| Saving submission...')
    submission = pd.read_csv('../data/sample_submission.csv')
    submission["prediction"] = preds
    submission.to_csv("submission.csv", index=False)    

