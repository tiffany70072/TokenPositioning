# TokenPositioning
Explain the capability of the token-positioning task
* [SPEC link](https://docs.google.com/document/d/1bZbSScbywq1Tcj9qWRZXh08b6bYLAT8gqrcwni744Lk/edit#heading=h.uc1nrkgjomy8)


## How to train
1. 在 src 底下，run ```python3 main.py --task=autoenc-last --units=32 --max_epochs=1000 --mode=train --data_name=fake-data --batch_size=8```
2. ps: 上面是 toy data 的例子，因為只有八筆資料，所以要很大的 max_epoch，你平常跑 max_epochs 應該設一百以內就夠？
3. 如果 training 正常的話，會出現兩個檔案：（跟 SPEC 寫得一樣）
    * saved_model/xxx/seq2seq.h5（這是 keras 的存法ＸＤ）
    * result/xxx/training_log.txt
    * xxx = {data_name}_units={units}_seed={seed}
4. training_log 裡面有三種 validation data 的正確率
    * each acc 是每個字分別算正確率
    * all 是整句對才算對
    * last 是這個 task 我特別多寫的，只會看 EOS 前一個字對不對，並確定兩邊的最後一個字出現在同個位子
5. 我生 fake-data 方法在 app/test-autoencoder-last.ipynb 裡面，可以看跟你想得有沒有一樣
6. 我的 fake-data 長這樣：（可以訓練到正確率 = 1）
```python
encoder_train = np.array([[1, 3, 3, 3, 3, 2],
                          [0, 1, 3, 3, 3, 2], 
                          [0, 0, 1, 3, 3, 2],
                          [0, 0, 0, 1, 3, 2],
                          [1, 4, 4, 4, 4, 2],
                          [0, 1, 4, 4, 4, 2],
                          [0, 0, 1, 4, 4, 2],
                          [0, 0, 0, 1, 4, 2]])
decoder_train = np.array([[1, 5, 5, 5, 3, 2],
                          [1, 5, 5, 3, 2, 0], 
                          [1, 5, 3, 2, 0, 0],
                          [1, 3, 2, 0, 0, 0],
                          [1, 5, 5, 5, 4, 2],
                          [1, 5, 5, 4, 2, 0],
                          [1, 5, 4, 2, 0, 0],
                          [1, 4, 2, 0, 0, 0]])
```

## Preprocess txt file 
```python
import collections
from keras.preprocessing.sequence import pad_sequences

common_text = ... # collections.Counter 找出 training_data 前 20000 常見的字，照頻率排序
src_token = ['PAD', 'SOS', 'EOS'] + common_text + ['OOV']
src_ctoi = dict((c, i) for i, c in enumerate(src_token)) 
src_itoc = dict((i, c) for i, c in enumerate(src_token))

src = ... # Map data from txt file to 2d list with type int
tgt = ...
# max_src_len = max_tgt_len = 13
src_padded = pad_sequences(src, maxlen=max_src_len, padding='pre', truncating='pre')  # -> 2d array
tgt_padded = pad_sequences(tgt, maxlen=max_tgt_len, padding='post', truncating='post')

```

## How to run neuron_selection
1. 在 src 底下，run ```python3 pipeline_neuron_selection.py --...``` 或是跑在 app 底下開 neuron_selection_0415.ipynb，兩個是一樣的東西
2. 需要設定 task, data_name, units, random_seed，找到你現在想要分析的是哪個存好的 model（所有值就跟訓練時設定一樣）
3. 需要設定 token, T_list，表示你現在想要分析的是哪個 token 跟哪些 T，例如 token = 3, T_list = [5, 7]，每跑一次只會分析一個 token 
4. 裡面會做幾件事
   1. 利用 RFE 找出 store neuron（每個 token，每個 T 的每個 time step 有自己一組）
   2. 利用 RFE 找出 counter neuron（每個 token，每個 T 有自己一組） 
   3. 利用 integrated gradients 找出 ig neuron（每個 token，每個 T 的每個 time step 有自己一組）
   4. 真正的 important store neuron 是 intersection of (store_neuron, ig_neuron), important counter neuron 類似
   5. Verify important store neuron: 把他們變成 0 正確率會變低 (disable)，只留下他們正確率要高（enable)
   6. Veirfy important counter neuron: 把他們換成前一個 state 的值，正確的字會晚一個 time step 出現！
 
   
