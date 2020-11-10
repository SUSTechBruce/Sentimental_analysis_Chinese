# Sentimental_analysis_Chinese
Chinese sentences sentimental analysis
#### Step1: Train model
run: - Bi_LSTM_model
     - TextCNN
     - LSTM_model
#### Step2: Inference
- Inference some chinese movies sentimental results in inference.py
```python
if __name__ == "__main__":
    sentence_negative = '这部电影演的太烂的，演员也烂，台词也烂，整个电影就是一个烂片'
    sentence_positive = '这部电影有点善良。'
    tensor = get_sentence_infer(sentence_positive)
    predict_sentence(tensor, 'Bi_LSTM')

```
#### 分词
结巴分词可能会快一些，清华和北大的分词load model需要一定时间
