import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split


PAD_id = 0
SOS_id = 1
EOS_id = 2
DUM_id = 3

class DataGenerator():

    def __init__(self, vocab_dist, min_len, max_len, len_dist, token_dist, folder_name, train_size, valid_size):
        
        assert type(min_len) is int and type(max_len) is int
        assert round(sum(vocab_dist), 10) == 1.0

        self.vocab_dist = vocab_dist
        self.min_len = min_len
        self.max_len = max_len
        self.len_dist = len_dist
        self.token_dist = token_dist
        self.folder_name = folder_name
        self.train_size = train_size
        self.valid_size = valid_size
        
        self.token_start_idx = 4
        
    def generate(self, random_state):
        
        np.random.seed(random_state)

        total_data_size = self.train_size + self.valid_size
        all_sent_num_dist = self.getLenSentNum(self.min_len, self.max_len, self.len_dist, total_data_size)

        all_token_sent_num_dist = []
        for sidx, size in enumerate(all_sent_num_dist):
            tmp = self.getLenSentNum(0, (self.min_len+sidx-1), self.token_dist, size) \
                + [ 0 for i in range(self.max_len - min_len - sidx) ]
            all_token_sent_num_dist.append(tmp)
        all_token_sent_num_dist = np.array(all_token_sent_num_dist).sum(axis=0).tolist()

        all_encode_sent = self.generate_sentences(all_sent_num_dist, self.vocab_dist, self.min_len)
        all_decode_sent = self.generateDecode(all_encode_sent)

        # post-processing ex: merge control signal to encode sent, add <SOS>, <EOS>, <PAD>
        all_data = self.post_processing(all_encode_sent, all_decode_sent)

        train_encode_data, valid_encode_data, train_decode_data, valid_decode_data = \
            train_test_split( all_data[0], all_data[1], test_size=valid_size, random_state=random_state)
        
        train_data = (train_encode_data, train_decode_data)
        valid_data = (valid_encode_data, valid_decode_data)

        return train_data, valid_data


    def getLenSentNum(self, min_len, max_len, len_dist, data_size):
        from collections import Counter

        len_diff = (max_len - min_len) + 1
        
        if len_dist == 'uniform':
            tmp = data_size
            len_sent_num = []
            for i in range(len_diff):
                if i == (len_diff-1):
                    len_sent_num.append(tmp)
                else:
                    len_sent_num.append( (data_size // len_diff) )
                    tmp -= (data_size // len_diff)

        elif len_dist == 'normal':
            base_size = int(data_size / len_diff) // len_diff
            mount_size = int(data_size * (len_diff - 1) / len_diff)
            
            half_diff = len_diff / 2
            tmp = np.random.normal(loc=0, scale=half_diff, size=mount_size).round(0)
            tmp = dict(Counter(tmp))
            len_sent_num = [ (base_size + tmp[i]) for i in range(-int(half_diff), int(half_diff)+1) ]
            
            import math
            rest_data_size = data_size - sum(len_sent_num)
            peak_idx_1 = math.ceil(half_diff)
            peak_idx_2 = math.floor(half_diff)
            
            len_sent_num[peak_idx_1] += math.floor(rest_data_size / 2)
            len_sent_num[peak_idx_2] += math.ceil(rest_data_size / 2)
        else:
            raise NotImplementedError()

        return len_sent_num

    def generate_sentences(self, sent_num_dist, vocab_dist, min_len):
        all_data = []
        for lidx in range(len(sent_num_dist)):
            data_num = sent_num_dist[lidx]
            sent_len = min_len + lidx
            data = np.random.choice([self.token_start_idx + idx for idx in range(len(vocab_dist))],\
                 size=(data_num, sent_len), p=vocab_dist).tolist()

            all_data += data
        return all_data


    def generateDecode(self, encode_sent):
        decode_sent = []
        for es in encode_sent:
            ds = [ DUM_id for i in range(len(es)-1) ] +  [es[-1] ]
            decode_sent.append(ds)
        
        return decode_sent

    # post-processing ex: merge control signal to encode sent, add <SOS>, <EOS>, <PAD>
    def padding(self, inp_sent, pad_token, max_len, pad_type="back"):
        assert pad_type == "back" or pad_type == "front"

        if len(inp_sent) < max_len:
            pad_num = max_len - len(inp_sent)
            
            if pad_type == 'back':
                return (inp_sent + [pad_token for i in range(pad_num)])
            else:
                return ([pad_token for i in range(pad_num)] + inp_sent)
        elif len(inp_sent) == max_len:
            return inp_sent

    def post_processing(self, encode_sent, decode_sent):
        
        for i in range(len(encode_sent)):
            # add EOS to encode sentence
            encode_sent[i] = [SOS_id] + encode_sent[i] + [EOS_id]
            # pad sentence
            encode_sent[i] = self.padding(encode_sent[i], PAD_id, \
                self.max_len + 2, "front")

        for i in range(len(decode_sent)):
            decode_sent[i] = [SOS_id] + decode_sent[i] + [EOS_id]
            decode_sent[i] = self.padding(decode_sent[i], PAD_id, self.max_len + 2, "back")

        return encode_sent, decode_sent


    def dumpFile(self, data_pair, out_folder, data_type="train"):

        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

        np.save( os.path.join(out_folder, "encoder_" + data_type + ".npy"), np.array(data_pair[0]) )
        np.save( os.path.join(out_folder, "decoder_" + data_type + ".npy"), np.array(data_pair[1]) )



if __name__ == '__main__':

    # parser = argparse.ArgumentParser()

    vocab_dist = [0.4, 0.3, 0.2, 0.1]
    min_len = 10
    max_len = 10
    len_dist = 'uniform'
    token_dist = 'uniform'
    folder_name = "./data/synthesis_uniform_uniform_autoencoder/"
    train_size = 50000
    valid_size = 20000

    dg1 = DataGenerator(vocab_dist, min_len, max_len, len_dist, token_dist, folder_name, train_size, valid_size)

    td1, vd1 = dg1.generate(random_state=20200414)

    dg1.dumpFile(td1, folder_name, data_type='train')
    dg1.dumpFile(vd1, folder_name, data_type='valid')
    
    vocab_dist = [0.4, 0.3, 0.2, 0.1]
    min_len = 1
    max_len = 13
    len_dist = 'normal'
    token_dist = 'uniform'
    folder_name = "./data/synthesis_normal_uniform_autoencoder/"
    train_size = 50000
    valid_size = 20000

    dg2 = DataGenerator(vocab_dist, min_len, max_len, len_dist, token_dist, folder_name, train_size, valid_size)

    td2, vd2 = dg2.generate(random_state=20200414)

    dg2.dumpFile(td2, folder_name, data_type='train')
    dg2.dumpFile(vd2, folder_name, data_type='valid')
    
