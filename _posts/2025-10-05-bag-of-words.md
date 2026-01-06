---
layout: post
title: Bag of Words
date: 2025-10-05
#image: https://placehold.it/900x300
lead: "In this notebook, you will find guidelines to download, prepare, and store the Bag of Words Data Set from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml)."
# categories:
# - NLP-basics
# - python
categories: [Reinforcement Learning]
subtitle: Learn more about NLP basics
---

The post is about how to download the data and start exploring it.

# Download the data

Follow these guidelines to download the data:

- Visit [the UCI website](https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/)
- Click on **docword.enron.txt.gz** to download the data.
- Unzip the data and save it in the same folder that contains this notebook.
- Then click on **vocab.enron.txt** to download the word names.
- Save vocab.enron.txt in the same folder that contains this notebook.

You can find more information about this particular dataset [here](https://archive.ics.uci.edu/ml/datasets/Bag+of+Words).

$$L(\phi_i) = \mathbb{E}_{(s,a,r,s') \in D} \left[Q_{\phi_i}(s,a) - r - \max_{a'} Q_{\phi'_i}(s',a') \right]^2$$

$$
\begin{align}
J(\theta) &= \mathbb{E}\Big[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \Big] \label{eq:objective} \\
&= V^\pi(s_0)
\end{align}
$$

```python
import pandas as pd
```

```python
# load the word counts

data = pd.read_csv("docword.enron.txt", sep=" ", skiprows=3, header=None)
data.columns = ["docID", "wordID", "count"]

data.head()
```

| Tables   |      Are      |  Cool |
| -------- | :-----------: | ----: |
| col 1 is | left-aligned  | $1600 |
| col 2 is |   centered    |   $12 |
| col 3 is | right-aligned |    $1 |

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>docID</th>
      <th>wordID</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>118</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>285</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1229</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1688</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2068</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
# load the words

words = pd.read_csv("vocab.enron.txt", header=None)
words.columns = ["words"]

words.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aaa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aaas</td>
    </tr>
    <tr>
      <th>2</th>
      <td>aactive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aadvantage</td>
    </tr>
    <tr>
      <th>4</th>
      <td>aaker</td>
    </tr>
  </tbody>
</table>
</div>

```python
# select at random 10 words

words = words.sample(10, random_state=290917)

words
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8704</th>
      <td>eurobond</td>
    </tr>
    <tr>
      <th>13618</th>
      <td>keen</td>
    </tr>
    <tr>
      <th>11114</th>
      <td>halligan</td>
    </tr>
    <tr>
      <th>19968</th>
      <td>pvr</td>
    </tr>
    <tr>
      <th>23327</th>
      <td>soda</td>
    </tr>
    <tr>
      <th>20714</th>
      <td>refundable</td>
    </tr>
    <tr>
      <th>390</th>
      <td>advice</td>
    </tr>
    <tr>
      <th>6257</th>
      <td>decker</td>
    </tr>
    <tr>
      <th>8680</th>
      <td>etis</td>
    </tr>
    <tr>
      <th>3370</th>
      <td>cab</td>
    </tr>
  </tbody>
</table>
</div>

```python
data = words.merge(data, left_index=True, right_on="wordID")

data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>words</th>
      <th>docID</th>
      <th>wordID</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>137715</th>
      <td>eurobond</td>
      <td>2021</td>
      <td>8704</td>
      <td>2</td>
    </tr>
    <tr>
      <th>140167</th>
      <td>eurobond</td>
      <td>2050</td>
      <td>8704</td>
      <td>11</td>
    </tr>
    <tr>
      <th>151530</th>
      <td>eurobond</td>
      <td>2269</td>
      <td>8704</td>
      <td>2</td>
    </tr>
    <tr>
      <th>155066</th>
      <td>eurobond</td>
      <td>2352</td>
      <td>8704</td>
      <td>2</td>
    </tr>
    <tr>
      <th>156247</th>
      <td>eurobond</td>
      <td>2375</td>
      <td>8704</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

```python
# reconstitute the bag of words dataset

bow = data.pivot(index="docID", columns="words", values="count")
bow.fillna(0, inplace=True)
bow.reset_index(inplace=True, drop=True)
bow.shape
```

    (1388, 10)

```python
bow.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>words</th>
      <th>advice</th>
      <th>cab</th>
      <th>decker</th>
      <th>etis</th>
      <th>eurobond</th>
      <th>halligan</th>
      <th>keen</th>
      <th>pvr</th>
      <th>refundable</th>
      <th>soda</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
bow.to_csv("../bag_of_words.csv", index=False)
```

```python

```
