# Sentiment Analysis on Movie Reviews

## მონაცემები

ტრეინ დატაში გვაქვს 156060 მონაცემი.
ასევე ხდება ასეთი რამ, რომ წინადადება იჭრება და სხვადასხვა სემპლად ვიყენებთ შემდეგ.

ზოგადად ტრეინის მონაცემები ესე გამოიყურება:

```
count    156060.000000
mean          7.203364
std           7.024604
min           0.000000
25%           2.000000
50%           5.000000
75%          10.000000
max          52.000000
```
<img width="1390" height="489" alt="download" src="https://github.com/user-attachments/assets/21089fec-a272-479b-b6ed-6196e91b294e" />

ასევე ამ plot-ის მიხედვით ყველაზე მეტი გვაქვს ნეიტრალური მონაცემი, ამიტომ ცოტა არადაბალანსებული დატა გვაქვს.

---

## პრეპროცესინგი

დავიწყე შემდეგ პრეპროცესინგი და ჯერ გავწმინდე ტექსტი `clean_text` მეთოდით — დავაპატარავე ყველა (lower case), ასევე რამდენიმე მცირე ოპერაციაც ჩავუტარე.

ამის შემდეგ ვაკეთებ encoding-ს — რაც გულისხმობს, რომ ყველა წინადადება გადამყავს 50 სიგრძის ვექტორად. მაგალითად:

```
"the film was great" → ["the", "film", "was", "great"] → [4, 27, 89, 134, 0, 0, ..., 0]
```

სიტყვა რომ vocab-ში არ მოიძებნება, ჩაიწერება `1` (`<UNK>`). მოკლე ფრაზები ბოლოდან ივსება `0`-ებით (`<PAD>`), რათა ყველა tensor ერთი ზომის იყოს.

Vocabulary size: **16198** (სიტყვები რომლებიც მინიმუმ 2-ჯერ გვხვდება train-ში)

---

## ექსპერიმენტი 1 — RNN Block

პირველ რიგში გავტესტე მარტივი RNN არქიტექტურა (`nn.RNN`), `tanh`-ის ნაცვლად `relu` გამოვიყენე - vanishing gradient პრობლემის ის გამო.

არქიტექტურა:
- `Embedding(16198, 128)`
- `RNN` — 1 ფენა, hidden size 256, `nonlinearity='relu'`
- `Linear(256, 5)`

```
Epoch 01 | Train Loss: 1.2933 Acc: 0.5091 | Val Loss: 1.2840 Acc: 0.5099
Epoch 02 | Train Loss: 1.2998 Acc: 0.5096 | Val Loss: 1.2847 Acc: 0.5099
Epoch 03 | Train Loss: 1.2859 Acc: 0.5100 | Val Loss: 1.2878 Acc: 0.5099
Epoch 04 | Train Loss: 1.2868 Acc: 0.5099 | Val Loss: 1.2838 Acc: 0.5099
Epoch 05 | Train Loss: 1.2853 Acc: 0.5100 | Val Loss: 1.2837 Acc: 0.5099
Epoch 06 | Train Loss: 1.2849 Acc: 0.5094 | Val Loss: 1.2464 Acc: 0.5118
Epoch 07 | Train Loss: 1.2077 Acc: 0.5189 | Val Loss: 1.1551 Acc: 0.5407
Epoch 08 | Train Loss: 1.1390 Acc: 0.5467 | Val Loss: 1.1170 Acc: 0.5546
Epoch 09 | Train Loss: 1.0880 Acc: 0.5645 | Val Loss: 1.0671 Acc: 0.5777
Epoch 10 | Train Loss: 1.0544 Acc: 0.5764 | Val Loss: 1.0440 Acc: 0.5791

Best Val Accuracy: 0.5791
```

პირველი 5 epoch-ი პრაქტიკულად გაჩერებული იყო, მე-6-ზე დაიწყო სწავლა . Kaggle public score: **0.57204**

---

## ექსპერიმენტი 2 — BiLSTM

RNN Block-ის შედეგით კმაყოფილი არ ვიყავი, ამიტომ გადავედი **Bidirectional LSTM**-ზე. სენტიმენტის გასაგებად ხშირად საჭიროა კონტექსტი ორივე მიმართულებიდან, მაგ. "not entirely bad"-ში "not" გავლენას ახდენს სიტყვებზე რომლებიც მას მოსდევს, ამავდროულად "bad"-ი ცვლის ადრინდელი კონტექსტის გაგებას. BiLSTM ამ ორივეს ერთდროულად ითვალისწინებს.

არქიტექტურა:
- `Embedding(16198, 128)` — ყოველი სიტყვა → 128-განზომილებიანი ვექტორი
- `Bidirectional LSTM` — 2 ფენა, hidden size 256, dropout 0.3
- `Linear(512, 5)` — საბოლოო კლასიფიკატორი (512 = 256 × 2 მიმართულება)


### ტრენინგი

- Optimizer: Adam, lr=1e-3
- Loss: CrossEntropyLoss
- Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)
- Gradient clipping: 1.0
- Batch size: 128
- Epochs: 10

```
Epoch 01 | Train Loss: 1.1202 Acc: 0.5521 | Val Loss: 0.9914 Acc: 0.6053
Epoch 02 | Train Loss: 0.9441 Acc: 0.6163 | Val Loss: 0.8865 Acc: 0.6450
Epoch 03 | Train Loss: 0.8590 Acc: 0.6514 | Val Loss: 0.8302 Acc: 0.6637
Epoch 04 | Train Loss: 0.8032 Acc: 0.6711 | Val Loss: 0.8124 Acc: 0.6731
Epoch 05 | Train Loss: 0.7651 Acc: 0.6865 | Val Loss: 0.7970 Acc: 0.6780
Epoch 06 | Train Loss: 0.7346 Acc: 0.6992 | Val Loss: 0.7869 Acc: 0.6811
Epoch 07 | Train Loss: 0.7076 Acc: 0.7084 | Val Loss: 0.8020 Acc: 0.6787
Epoch 08 | Train Loss: 0.6879 Acc: 0.7165 | Val Loss: 0.7919 Acc: 0.6861
Epoch 09 | Train Loss: 0.6682 Acc: 0.7236 | Val Loss: 0.7983 Acc: 0.6818
Epoch 10 | Train Loss: 0.6330 Acc: 0.7389 | Val Loss: 0.8087 Acc: 0.6866

Best Val Accuracy: 0.6866
```

6-7 epoch-ის შემდეგ val loss-მა გაჩერება დაიწყო (overfitting-ის ნიშანი), ამიტომ მეტი epoch-ი სარგებელს არ მოიტანდა.

### ვალიდაციის შედეგები

```
              precision    recall  f1-score   support

    Negative       0.50      0.39      0.44       707
Somewhat Neg       0.59      0.56      0.57      2727
     Neutral       0.76      0.83      0.79      7958
Somewhat Pos       0.62      0.57      0.59      3293
    Positive       0.58      0.51      0.55       921

    accuracy                           0.69     15606
   macro avg       0.61      0.57      0.59     15606
weighted avg       0.68      0.69      0.68     15606
```

საერთო სიზუსტე: **69%**

ყველაზე კარგი შედეგი გვაქვს Neutral კლასზე (F1=0.79) — რაც ლოგიკურია, რადგან ის ყველაზე მეტი სემპლი გვაქ ამაში. 

შემდეგ კი ტესტზე აჩვენა ეს შედეგი (kaggle-ზე): **0.65083**

---

## შედარება

| მოდელი | Val Acc | Kaggle Score |
|---|---|---|
| RNN Block | 0.5791 | 0.57204 |
| BiLSTM | 0.6866 | 0.65083 |
