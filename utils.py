import numpy as np
import torch 
from torch import nn
from torch.nn.utils.rnn import pad_sequence

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, CharTensor, LongTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor, CharTensor, LongTensor

from transformers import BertTokenizer
from sklearn import metrics

from sklearn.metrics import classification_report
from torch.nn.functional import one_hot, binary_cross_entropy


def match_seq_len(q_seqs, r_seqs, seq_len, pad_val=-1):
    '''
        Args: 
            q_seqs: the question(KC) sequence with the size of \
                [batch_size, some_sequence_length]
            r_seqs: the response sequence with the size of \
                [batch_size, some_sequence_length]

            Note that the "some_sequence_length" is not uniform over \
                the whole batch of q_seqs and r_seqs
            
            seq_len: the sequence length to match the q_seqs, r_seqs \
                to same length
            pad_val: the padding value for the sequence with the length \
                longer than seq_len

        Returns:
            proc_q_seqs: the processed q_seqs with the size of \
                [batch_size, seq_len + 1]
            proc_r_seqs: the processed r_seqs with the size of \
                [batch_size, seq_len + 1]
    '''

    proc_q_seqs = []
    proc_r_seqs = []

    # seq_len은 q_seqs와 r_seqs를 같은 길이로 매치하는 시퀀스 길이를 의미함.
    # q_seq는 유저의 스킬에 대한 인덱스 리스트를 갖는 리스트임.
    # 주어진 q, r시퀀스들을 seq_len 만큼 자르는 것이라고 보면 됨
    for q_seq, r_seq in zip(q_seqs, r_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq): # i + seq_len + 1 이 주어진 문제 집합보다 길이가 작을 때, e.g.) 0 + 100 + 1 < 128
            proc_q_seqs.append(q_seq[i:i + seq_len + 1]) # i부터 i + seq_len + 1 범위의 elements를 퀘스천 시퀀스에 넣음 e.g.) 0부터 0 + 100 + 1 원소의 배열 시퀀스를 proc_q에 할당함
            proc_r_seqs.append(r_seq[i:i + seq_len + 1]) # 위와 동일. e.g.) 0부터 0 + 100 + 1 원소 배열 시퀀스를 proc_r에 할당함

            i += seq_len + 1 # i에 seq_len + 1을 더하여 len(q_seq)보다 크게 만듬

        # seq_len 만큼 자른 sequence들을 padding값이 들어가 있는 합쳐서 넣음
        # 자른 건 길이 모자라니깐 padding 값으로 대체해서 넣음, 아닐 경우 원래 시퀀스에 패딩값이 들어있는 배열 붙여넣음
        proc_q_seqs.append(
            np.concatenate([
                q_seq[i:],
                np.array([pad_val] * (i + seq_len + 1 - len(q_seq))) # padding value가 들어있는 배열의 원소를 * 갯수만큼 생성함 (여기선 0, seq_len q_seq가 128이라 가정하면 129 - 128, 즉 1개만 만듬)
            ]) 
        )
        proc_r_seqs.append(
            np.concatenate([
                r_seq[i:],
                np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
            ])
        )
        # 마지막 1개의 원소들은 패딩해서 넣게 됨

    return proc_q_seqs, proc_r_seqs


def collate_fn(batch, pad_val=-1):
    '''
    This function for torch.utils.data.DataLoader

    Returns:
        q_seqs: the question(KC) sequences with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
        r_seqs: the response sequences with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
        qshft_seqs: the question(KC) sequences which were shifted \
            one step to the right with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
        rshft_seqs: the response sequences which were shifted \
            one step to the right with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
        mask_seqs: the mask sequences indicating where \
            the padded entry is with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
    '''

    q_seqs = []
    r_seqs = []
    qshft_seqs = []
    rshft_seqs = []


    # q_seq와 r_seq는 마지막 전까지만 가져옴 (마지막은 padding value)
    # q_shft와 rshft는 처음 값 이후 가져옴 (우측 시프트 값이므로..)
    for q_seq, r_seq in batch:
        q_seqs.append(FloatTensor(q_seq[:-1])) 
        r_seqs.append(FloatTensor(r_seq[:-1]))
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))

    # pad_sequence, 첫번째 인자는 sequence, 두번째는 batch_size가 첫 번째로 인자로 오게 하는 것이고, 3번째 인자의 경우 padding된 요소의 값
    # 시퀀스 내 가장 길이가 긴 시퀀스를 기준으로 padding이 됨, 길이가 안맞는 부분은 늘려서 padding_value 값으로 채워줌
    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )


    # 마스킹 시퀀스 생성 
    # 일반 question 시퀀스: 패딩 밸류와 다른 값들은 모두 1로 처리, 패딩 처리된 값들은 0으로 처리.
    # 일반 question padding 시퀀스: 한 칸 옆으로 시프팅 된 시퀀스 값들이 패딩 값과 다를 경우 1로 처리, 패딩 처리 된 값들은 0으로 처리.
    # 마스킹 시퀀스: 패딩 처리 된 시퀀스 밸류들은 모두 0, 두 값 모두 패딩처리 되지 않았을 경우 1로 처리. (원본 시퀀스와 shift 시퀀스 모두의 값)
    # 예를 들어, 현재 값과 다음 값이 패딩 값이 아닐 경우 1, 현재 값과 다음 값 둘 중 하나라도 패딩일 경우 0으로 처리함.
    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    # 원본 값의 다음 값이(shift value) 패딩이기만 해도 마스킹 시퀀스에 의해 값이 0로 변함. 아닐경우 원본 시퀀스 데이터를 가짐.
    q_seqs, r_seqs, qshft_seqs, rshft_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs
    

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs