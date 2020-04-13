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

