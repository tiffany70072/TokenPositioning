# TokenPositioning
Explain the capability of the token-positioning task
* [SPEC link](https://docs.google.com/document/d/1bZbSScbywq1Tcj9qWRZXh08b6bYLAT8gqrcwni744Lk/edit#heading=h.uc1nrkgjomy8)


## How to train
在 src 底下，run（這是一個 toy data 的例子）
```python3 main.py --task=autoenc-last --units=32 --max_epochs=1000 --mode=train --data_name=fake-data --batch_size=8```
(ps: 因為資料筆數只有八筆，所以我設了很大的 max_epoch，你資料筆數多的話，max_epochs 應該設一百以內？）


1. 如果 training 有成功的話
會在 saved_model 裡面看到新存好的 model
還有在 result 裡面會看到一個 training_log
（跟 SPEC 寫得一樣）

2. 我的 fake-data 的長相放在 app/test-autoencoder-last.ipynb 裡面
可以看跟你想得有沒有一樣

3. training_log 裡面會有三種 validation data 的正確率
  * each acc 是每個字分別算正確率
  * all 是整句對才算對
  * last 是這個 task 我特別多寫的，只會看 EOS 前一個字對不對，並確定兩邊的最後一個字出現在同個位子
