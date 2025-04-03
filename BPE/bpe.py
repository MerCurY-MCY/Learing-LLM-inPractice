import re

from collections import Counter
import toolz  # 处理迭代器、字典、列表等数据结构的工具


def wordpunct_tokenize(text):
    # \w匹配Unicode字符     ^\w\s匹配非Unicode字符和非空白字符（标点符号）

    _pattern = r"\w+|[^\w\s]+"

    # 编译为对象   re.MULTILINE表示$和^可以匹配下一行开头和结尾，re.DOTALL表示点号也可以匹配换行符
    _regexp = re.compile(_pattern, flags=re.UNICODE | re.MULTILINE | re.DOTALL)
    return _regexp.findall(text)


corpus = ["Baby, I don't feel so good",
          "six words you never understood",
          "I'll never let you go",
          "five words you'll never say (aww)",
          "I laugh along like nothing's wrong",
          "four days has never felt so long",
          "If three's a crowd and two was us",
          "one slipped away (hahahahaha)",
          "I just wanna make you feel okay",
          "But all you do is look the other way",
          "I can't tell you how much I wish I didn't wanna stay",
          "I just kinda wish you were gay",
          "Is there a reason we're not through?",
          "Is there a 12-step just for you?",
          "Our conversation's all in blue",
          "11 \"heys\" (Hey, hey, hey, hey)",
          "Ten fingers tearin' out my hair",
          "Nine times, you never made it there",
          "I ate alone at seven, you were six minutes away",
          "How am I supposed to make you feel okay",
          "When all you do is walk the other way?",
          "I can't tell you how much I wish I didn't wanna stay",
          "I just kinda wish you were gay",
          "To spare my pride",
          "To give your lack of interest, an explanation",
          "Don't say I'm not your type",
          "Just say that I'm not your preferred sexual orientation",
          "I'm so selfish",
          "But you make me feel helpless, yeah",
          "And I can't stand another day",
          "Stand another day",
          "I just wanna make you feel okay",
          "But all you do is look the other way, hmm",
          "I can't tell you how much I wish I didn't wanna stay",
          "I just kinda wish you were gay",
          "I just kinda wish you were gay",
          "I just kinda wish you were gay",
          ]


class BPEtokenizer():
    special = ['<UKN>', '<END>', '<PAD>', '<MAD>']  # 对序列的特殊填充

    def __init__(self, vocab_size=10000, lowercase=True, basic_tokenizer=wordpunct_tokenize, user_specials=None):
        self.vocal_size = vocab_size
        self.lowercase = lowercase
        self.basic_tokenizer = basic_tokenizer
        self.special = dict(unk='<UNK>', sep='<SEP>',
                            pad='<PAD>', cls='<CLS>', mask='<MASK>')

    def loadAndTransform(self, vocab_fn=None, vocab=None):
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = [l.strip() for l in open(vocab_fn, 'r')]
        vocab_len = len(self. vocab)
        self.voToid = {x: y for x, y in enumerate(self.vocab)}  # 把字符转换为索引
        self.idTovo = {y: x for x, y in self.voto2d.items()}  # 把索引转换为字符

    def train(self, corpus: list, max_step=10000, out_fn='./BPE/vocabulary.txt'):

        ######################################### 统计词频################################################
        if self.lowercase:
            corpus = [s.lower() for s in corpus]

        # map用于把一个函数依次对一个数据结构中的元素使用，并返回一个迭代器
        corpus = list(map(self.basic_tokenizer, corpus))

        # 展平该列表
        corpus = toolz.concat(corpus)

        # 把每个元素转换成元组并加入结尾符，计算每个单词出现的次数。Counter返回一个元素计数的字典。
        split_corpus = Counter(tuple(word) + ('/w',) for word in corpus)

        # split_corpus = Counter([tuple(word)+ ('<\W>', ) for word in toolz.concat(map(self.basic_tokenizer, corpus))])

        ########################################## 逐步合并高频词为token并生成词表#################################
        vocab = self._count_vocab(split_corpus)

        for i in range(max_step):
            split_corpus, vocab_cnt = self._countAndMerge(
                split_corpus)  # 保留单词结构，统计一个单词内出现的二元子词的词频
            vocab = self._count_vocab(split_corpus)  # 把单词切碎，只统计字符级别的词频
            if len(vocab) > self.vocal_size or vocab_cnt < 0:
                break

        ########## 插入特殊词######################
        for s in self.special:
            if s not in vocab:
                vocab.insert(0, (s, 9999))

        ##### 导出列表#####

        with open(out_fn, 'w') as f:
            f.write('\n'.join(token for token, _ in vocab))

        self.vocab = [token for token, _ in vocab]

        return vocab

    def _count_vocab(self, split_corpus):
        _countWord = Counter([data for data in toolz.concat(
            [w * x for w, x in split_corpus.items()])])  # .items()顺序访问字典中的元素，enumerate访问元组或者列表
        _sortWord = sorted(_countWord.items(),
                           key=lambda x: x[1], reverse=True)  # 按第一维度降序排序字符
        return _sortWord

    def _countAndMerge(self, split_corpus):
        ngram = 2
        bigramCounter = Counter()

        for token, count in split_corpus.items():  # 循环扫描每个单词和其出现频率
            if count < 2:
                continue  # 跳过小于2的子词

            # 使用2的滑动窗口在单词上滚动
            for subwords in toolz.sliding_window(ngram, token):
                bigramCounter[subwords] += count  # 将每个长度为2的子词的出现次数记录下来

        if len(bigramCounter) > 0:
            # 找出最大频率的二元子词，max会循环读取可迭代对象，并对每个对象执行key对应的函数，比较函数计算出的值。（对于字典，读出的是键）
            max_bigram_key = max(bigramCounter, key=bigramCounter.get)
        else:
            return split_corpus, -1

        max_bigram_cnt = bigramCounter.get(max_bigram_key)

        list_split_corpus_key = list(split_corpus.keys())
        for tokens in list_split_corpus_key:

            # jion方法可以把数据结构中的参数合并成字符串，.前面的符号是合并时每个元素间插入的字符
            temp_tokens = ' '.join(tokens)

            temp_tokens = temp_tokens.replace(' '.join(max_bigram_key), ''.join(
                max_bigram_key))  # 把原始的token中的分离字符替换为合并在一起的二元高频字符

            # split方法通过括号里的字符把字符串分开，返回一个列表，再转换成元组得到例如(I lo v e) 的形式，其中lo是之前统计得到的高频二元字词
            new_tokens = tuple(temp_tokens.split(' '))

            # temp_split_corpus = tuple(' '.join(tokens).replace(' '.join(list_split_corpus_key), ''.join(list_split_corpus_key)).split(' '))
            if new_tokens != tokens:
                split_corpus[new_tokens] = split_corpus[tokens]
                split_corpus.pop(tokens)
        return split_corpus, max_bigram_cnt

    def tokenize(self, text, pre=None, mid=None, post='</w>'):
        all_tokens = []

        text = ''.join(text).strip()

        if self.lowercase:
            text = text.lower()
        split_text = list(self.basic_tokenizer(text))

        for tokens in split_text:

            tokens = list(tokens)  # 把词列表化以使用切片操作
            print(tokens)
            if pre:
                tokens = [pre] + tokens
            if post:
                tokens = tokens + [post]

            begin, end = 0, len(tokens)
            ######################### 贪心算法匹配序列里的词是否在训练出的词表里，从而把序列分割成token############
            while begin < end:
                sub_token = ''.join(tokens[begin:end])
                if begin > 0 and mid:
                    sub_token = mid + sub_token
                if begin > 0 and sub_token in self.vocab:  # 如果切片到的词在词表里，就存入token列表并把begin指针移动到end处，让end指向序列尾
                    all_tokens.append(sub_token)
                    begin = end
                    end = len(tokens)
                elif end - begin == 1:  # 如果序列长度为1还没匹配到，就标注未知词
                    all_tokens.append(self.special['unk'])
                    begin = end
                    end = len(tokens)
                else:
                    end -= 1  # 尾部指针向头部指针移动
        return all_tokens

    def _tokentoid(self, token):
        if token in self. vocab:
            return self.vocab.index(token)
        else:
            return self.vocab.index(self.unk)

    def _idtoToken(self, id):
        return self.vocab[id]

    def encode(self, text):  # 编码文本
        tokens = self.tokenize(text)
        idforTokens = [list(map(self._idtoToken, token) for token in tokens)]
        return idforTokens

    def decode(self, text_ids):   # 解码文本
        text = []
        for token_ids in text_ids:
            tokens = [list(map(self._idtoToken, token_id)
                           for token_id in token_ids)]
            tokens = ''.join(tokens).replace('/w', '')
            text.append[tokens]
        return text


BPE = BPEtokenizer()
vocab = BPE.train(corpus=corpus)
print(vocab)
test_text = [
    'White shirt now red, my bloody nose',
    'Sleepin\', you\'re on your tippy toes',
    'Creepin\' around like no one knows',
    'Think you\'re so criminal',
    'Bruises on both my knees for you',
    'Don\'t say thank you or please',
    'I do what I want when I\'m wanting to',
    'My soul so cynical',
    'So you\'re a tough guy',
    'Like it really rough guy',
    'Just can\'t get enough guy',
    'Chest always so puffed guy',
    'I\'m that bad type',
    'Make your mama sad type',
    'Make your girlfriend mad type',
    'Might seduce your dad type',
    'I\'m the bad guy, duh',
    'I\'m the bad guy',
    'I like it when you take control',
    'Even if you know that you don\'t own me',
    'I\'ll let you play the role',
    'I\'ll be your animal',
    'My mommy likes to sing along with me',
    'But she won\'t sing this song',
    'If she reads all the lyrics',
    'She\'ll pity the men I know',
    'So you\'re a tough guy',
    'Like it really rough guy',
    'Just can\'t get enough guy',
    'Chest always so puffed guy',
    'I\'m that bad type',
    'Make your mama sad type',
    'Make your girlfriend mad type',
    'Might seduce your dad type',
    'I\'m the bad guy, duh',
    'I\'m the bad guy',
    'Duh',
    'I\'m only good at bein\' bad, bad',
    'I like when you get mad',
    'I guess I\'m pretty glad that you\'re alone',
    'You said she\'s scared of me?',
    'I mean, I don\'t see what she sees',
    'But maybe it\'s \'cause I\'m wearing your cologne',
    'I\'m the bad guy',
    'I\'m the bad guy',
    'Bad guy, bad guy',
    'I\'m the bad'
]

output_tokens = BPE.tokenize(test_text)
print(output_tokens)
with open('./BPE/output_tokens.txt', 'w') as f:
    f.write('\n'.join(output_tokens))
