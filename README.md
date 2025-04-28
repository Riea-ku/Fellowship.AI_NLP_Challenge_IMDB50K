# 🧠 IMDB Sentiment Analysis 


```markdown
# 🎬 Lights, Camera, Sentiment! 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Popcorn](https://img.shields.io/badge/-%F0%9F%8D%BF%20Movie%20Magic%20-blueviolet)

Ever cried at a movie only to realize the critic reviews called it "sentimental garbage"? Me too. That's why I built this **IMDB Review Sentiment Analyzer** - to understand why we feel what we feel about films.


### 🔍 What Makes This Special
- Human-like text understanding: Expands "OMG that plot twist!" into proper analysis
- Error analysis: Shows me exactly where it gets confused (we all have our blind spots)
- Visual storytelling: Because numbers alone can't capture the drama of sentiment

## 🎥 The Dataset Drama
Using the [IMDB 50K Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) - a balanced mix of:
- ⭐⭐⭐⭐⭐ "Masterpiece!" 
- ⭐ "I want my 2 hours back"

```python
# The script of our data story
class Config:
    MAX_TOKENS = 20000    # How many words we consider "vocabulary"
    BATCH_SIZE = 64       # Our attention span for learning
    EPOCHS = 15           # How many times we watch the same movie
    TEST_SIZE = 0.2       # The critic vs audience split
```

## 🎬 Director's Cut (Model Architecture)

```python
# Act 1: The Setup
Text Input → TF-IDF Vectorization 

# Act 2: The Conflict  
→ Dense(256, ReLU) + Dropout(0.5) + BatchNorm

# Act 3: The Resolution  
→ Dense(128, ReLU) + Dropout(0.3) 

# Finale: The Verdict
→ Output(Sigmoid) # Thumbs up or down?
```

**Training Montage:**
- Optimizer: Adam (learning rate: 0.001 - not too fast, not too slow)
- Loss Function: Binary Crossentropy (how sad when wrong)
- Callbacks: Early Stopping (knows when to walk away)

## 🏆 Performance Review
After training through 7 dramatic epochs (complete with plot twists in validation metrics), our model achieved:

|  Metric   | Peak Train | Val Score  | Critic's Notes                     |
|-----------|------------|------------|------------------------------------|
| Accuracy  | 0.9029     | 0.8910     | "Would watch again"                |
| Precision | 0.8972     | 0.8948     | "Spotlights truth"                 |
| Recall    | 0.9115     | 0.8877     | "Occasionally misses the subtext"  |
| AUC       | 0.9680     | 0.9575     | "Masterclass in separation"        |


The Director's Cut (Key Insights):

    Best Performance: Epoch 2 (val_accuracy: 0.891) - Our model's "Oscar moment"

    Most Dramatic Swing: Recall varied wildly from 0.5974 to 0.9612 - method acting at its finest

    Final Act: Settled at 89.1% accuracy - enough to trust its reviews more than some YouTube critics

## 🎞️ Running the Show

```bash
git clone https://github.com/Riea-ku/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
# Cue the training montage music...
code.py
```

## 🎭 Behind-the-Scenes Footage
- Finds patterns in negative reviews like "waste of time" vs "disappointing"
- Learns that "so bad it's good" requires cultural context
- Still gets fooled by sarcasm sometimes (don't we all?)

## 📜 The Fine Print
MIT Licensed - because art should be shared. Just don't use it to argue about Marvel movies at parties.



