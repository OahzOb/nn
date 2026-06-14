

class ChineseTokenizer:
    def __init__(self, corpus: str):
        corpus_dict = set(corpus)
        self.ch2id: dict = {'<unk>': 0}
        self.id2ch: dict = {0: '<unk>'}
        for i, ch in enumerate(corpus_dict, start=1):
            self.ch2id[ch] = i
            self.id2ch[i] = ch
        self.vocab_size = len(self.ch2id)

    def encode(self, ch_list: list) -> list[int]:
        id_list = [self.ch2id.get(ch, 0) for ch in ch_list]
        return id_list

    def decode(self, id_list: list) -> list[str]:
        ch_list = [self.id2ch.get(id, '<unk>') for id in id_list]
        return ch_list


if __name__ == '__main__':
    with open('datasets/红楼梦.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = ChineseTokenizer(corpus=text)
    print(f"Length of dict: {len(tokenizer.ch2id)}")
    ch_list = list(text[:20])
    ch_list.append('unk_test')
    id_list = tokenizer.encode(ch_list)
    ch_list_back = tokenizer.decode(id_list)
    print(ch_list)
    print(id_list)
    print(ch_list_back)