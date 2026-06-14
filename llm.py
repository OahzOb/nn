import random
import math

with open('datasets/红楼梦.txt', 'r', encoding='utf-8') as f:
    TXT = f.read()

class ChineseTokenizer:
    def __init__(self, corpus: str):
        corpus_dict = set(corpus)
        self.ch2id: dict = {'<unk>': 0}
        self.id2ch: dict = {0: '<unk>'}
        for i, ch in enumerate(corpus_dict, start=1):
            self.ch2id[ch] = i
            self.id2ch[i] = ch
        self.vocab_size = len(self.ch2id)
        print(f"Tokenizer with vocab_size {self.vocab_size}.")

    def encode(self, ch_list: list) -> list[int]:
        id_list = [self.ch2id.get(ch, 0) for ch in ch_list]
        return id_list

    def decode(self, id_list: list) -> list[str]:
        ch_list = [self.id2ch.get(id, '<unk>') for id in id_list]
        return ch_list


def _llm_forward():
    print('-' * 10, 'LLM forward 1 batch', '-' * 10)
    def _check_dim(mtx: list) -> list:
        dims = [len(mtx)]
        if isinstance(mtx[0], list):
            dims += _check_dim(mtx[0])
        return dims
    # params -----
    hidden_size = 12
    tokenizer = ChineseTokenizer(corpus=TXT)
    max_len_seq = 20
    n_heads = 4
    input_size = 10
    input_str = TXT[:input_size]
    print(f"Input string:\n{input_str}")
    input = list(input_str)
    # id embedding -----
    id_list = tokenizer.encode(ch_list=input)
    print(f"Id:\n{id_list}")
    id_embedding_mtx = [[random.random() for _ in range(hidden_size)] for _ in range(tokenizer.vocab_size)]
    id_embed = [id_embedding_mtx[id] for id in id_list]
    # position embedding -----
    position_embedding_mtx = [[random.random() for _ in range(hidden_size)] for _ in range(max_len_seq)]
    pos_embed = [position_embedding_mtx[idx] for idx, _ in enumerate(id_list)]
    # id + pos -----
    input_embed = [[ie + pe for ie, pe in zip(id_vec, pos_vec)] for id_vec, pos_vec in zip(id_embed, pos_embed)]
    # norm -----
    input_embed_norm = list()
    for vec in input_embed:
        vec_sum = sum(vec)
        vec_new = [elem / vec_sum for elem in vec]
        input_embed_norm.append(vec_new)
    # q, k, v weights -----
    weight_q = [[random.random() for _ in range(hidden_size)] for _ in range(hidden_size)]
    weight_k = [[random.random() for _ in range(hidden_size)] for _ in range(hidden_size)]
    weight_v = [[random.random() for _ in range(hidden_size)] for _ in range(hidden_size)]
    d_head = int(hidden_size // n_heads)
    weight_q_heads = [[vec for vec in weight_q[n * d_head: (n + 1) * d_head]] for n in range(n_heads)]
    weight_k_heads = [[vec for vec in weight_k[n * d_head: (n + 1) * d_head]] for n in range(n_heads)]
    weight_v_heads = [[vec for vec in weight_v[n * d_head: (n + 1) * d_head]] for n in range(n_heads)]
    # q, k, v values -----
    value_q_heads, value_k_heads, value_v_heads = list(), list(), list()
    value_heads_list = [value_q_heads, value_k_heads, value_v_heads]
    weight_heads_list = [weight_q_heads, weight_k_heads, weight_v_heads]
    for value_heads, weight_heads in zip(value_heads_list, weight_heads_list):
        for head in weight_heads:
            value_head = list()
            for vec in head:
                vec_new = list()
                for vec_input in input_embed_norm:
                    vec_new.append(sum([elem_head * elem_input for elem_head, elem_input in zip(vec, vec_input)]))
                value_head.append(vec_new)
            value_heads.append(value_head)
    # transpose -----
    def _transpose(mtx: list) -> list:
        mtx_transposed = [[0 for _ in range(len(mtx))] for _ in range(len(mtx[0]))]
        for i, vec in enumerate(mtx):
            for j, elem in enumerate(vec):
                mtx_transposed[j][i] = elem
        return mtx_transposed
    q_heads, k_heads, v_heads = list(), list(), list()
    for head in value_q_heads:
        q_head = _transpose(head)
        q_heads.append(q_head)
    for head in value_k_heads:
        k_head = _transpose(head)
        k_heads.append(k_head)
    v_heads = value_v_heads
    # attention -----
    def _softmax(vec: list) -> list:
        vec_e = [math.exp(elem) for elem in vec]
        e_sum = sum(vec_e)
        vec_softmax = [elem / e_sum for elem in vec_e]
        return vec_softmax
    mask = [[0 for _ in range(input_size)] for _ in range(input_size)]
    for i in range(len(mask)):
        for j in range(i + 1):
            mask[i][j] = 1
    scale_divider = math.sqrt(d_head)
    dropout_rate = 0.1
    attn_head = list()
    attn_out = list()
    attn_out_transposed = [[] for _ in range(input_size)]
    for q_head, k_head, v_head in zip(q_heads, k_heads, v_heads):
        attention_head = list()
        for q_vec in q_head:
            attn_vec = list()
            for k_vec in k_head:
                attn_score = sum([q_elem * k_elem for q_elem, k_elem in zip(q_vec, k_vec)]) / scale_divider
                attn_vec.append(attn_score)
            attention_head.append(attn_vec)

        attn_head_masked = [[-1e6 if mask_elem < 0.5 else attn_elem for attn_elem, mask_elem in zip(attn_vec, mask_vec)]
                            for attn_vec, mask_vec in zip(attention_head, mask)]
        attn_head = [_softmax(vec) for vec in attn_head_masked]
        mask_dropout = [[0 if random.random() < dropout_rate else 1 for _ in attn_vec] for attn_vec in attn_head]
        attn_head = [[attn_elem * mask_elem for attn_elem, mask_elem in zip(attn_vec, mask_vec)]
                     for attn_vec, mask_vec in zip(attn_head, mask_dropout)]
        attn_out = list()
        for attn_vec in attn_head:
            attn_out_vec = list()
            for v_vec in v_head:
                attn_out_elem = sum([attn_score * v_elem for attn_score, v_elem in zip(attn_vec, v_vec)])
                attn_out_vec.append(attn_out_elem)
            attn_out.append(attn_out_vec)

        for idx, attn_vec in enumerate(attn_out):
            attn_out_transposed[idx] += attn_vec
    # dropout -----
    attn_out_transposed = [[0 if random.random() < dropout_rate else attn_elem for attn_elem in attn_vec]
                           for attn_vec in attn_out_transposed]
    # residual connection -----
    input_attn = [[e1 + e2 for e1, e2 in zip(v1, v2)] for v1, v2 in zip(input_embed_norm, attn_out_transposed)]
    # ffn -----
    ffn_size = 16
    ffn_w1 = [[random.random() for _ in range(hidden_size)] for _ in range(ffn_size)]
    ffn_w2 = [[random.random() for _ in range(ffn_size)] for _ in range(hidden_size)]
    ffn_w3 = [[random.random() for _ in range(hidden_size)] for _ in range(ffn_size)]
    w1_out = list()
    for w1_vec in ffn_w1:
        w1_out_vec = list()
        for input_vec in input_attn:
            ffn_elem = sum([w1_elem * input_elem for w1_elem, input_elem in zip(w1_vec, input_vec)])
            w1_out_vec.append(ffn_elem)
        w1_out.append(w1_out_vec)
    w3_out = list()
    for w3_vec in ffn_w3:
        w3_out_vec = list()
        for input_vec in input_attn:
            ffn_elem = sum([w1_elem * input_elem for w1_elem, input_elem in zip(w3_vec, input_vec)])
            w3_out_vec.append(ffn_elem)
        w3_out.append(w3_out_vec)
    w1_activated = [[0 if elem < 0 else elem for elem in vec] for vec in w1_out]
    w1_w3 = [[w1_elem * w3_elem for w1_elem, w3_elem in zip(w1_vec, w3_vec)] for w1_vec, w3_vec in zip(w1_activated, w3_out)]
    w13_transposed = _transpose(w1_w3)
    ffn_out = list()
    for w13_vec in w13_transposed:
        ffn_vec = list()
        for w2_vec in ffn_w2:
            ffn_elem = sum([w13_elem * w2_elem for w13_elem, w2_elem in zip(w13_vec, w2_vec)])
            ffn_vec.append(ffn_elem)
        ffn_out.append(ffn_vec)
    # transformer combine -----
    transformer_out = [[e1 + e2 for e1, e2 in zip(v1, v2)] for v1, v2 in zip(input_attn, ffn_out)]
    # norm -----
    trans_out_norm = list()
    for vec in transformer_out:
        vec_sum = sum(vec)
        vec_new = [elem / vec_sum for elem in vec]
        trans_out_norm.append(vec_new)
    # logits -----
    w_out = [[random.random() for _ in range(hidden_size)] for _ in range(tokenizer.vocab_size)]
    logits = list()
    for trans_vec in trans_out_norm:
        logits_vec = list()
        for w_vec in w_out:
            logits_elem = sum([e1 * e2 for e1, e2 in zip(w_vec, trans_vec)])
            logits_vec.append(logits_elem)
        logits.append(logits_vec)
    # final -----
    def _argmax(vec: list) -> tuple[int, int]:
        max_idx = 0
        max_val = vec[0]
        for idx, val in enumerate(vec):
            if val > max_val:
                max_val = val
                max_idx = idx
        return max_idx, max_val
    id_list = [_argmax(vec)[0] for vec in logits]
    ch_list = tokenizer.decode(id_list=id_list)
    output_string = ''.join(ch_list)
    print("Dim checks: ")
    dim_dict = {
        'Input string': _check_dim(input),
        'Id embeddings': _check_dim(id_embed),
        'Position embeddings': _check_dim(pos_embed),
        'Combined input embeddings': _check_dim(input_embed),
        'Normalized input embeddings': _check_dim(input_embed_norm),
        'q/k/v weights': _check_dim(weight_q),
        'multihead q/k/v weights': _check_dim(weight_q_heads),
        'multihead q/k/v values': _check_dim(value_q_heads),
        'transposed multihead q/k values': _check_dim(q_heads),
        'attention score per head': _check_dim(attn_head),
        'attention output per head': _check_dim(attn_out),
        'attention output': _check_dim(attn_out_transposed),
        'ffn w1/2/3 weight': [_check_dim(w) for w in [ffn_w1, ffn_w2, ffn_w3]],
        'ffn w1/3 output': _check_dim(w1_out),
        'ffn w1 * w3': _check_dim(w1_w3),
        'ffn output': _check_dim(ffn_out),
        'transformer output': _check_dim(trans_out_norm),
        'logits': _check_dim(logits),
    }
    for key, value in dim_dict.items():
        print(key, ':', value)
    print(f"Final output string: {output_string}")

if __name__ == '__main__':
    _llm_forward()