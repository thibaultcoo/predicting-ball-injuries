# Predicting basketball injuries using ML
*Soughati Kenza, Henry-Biabaud Briac, Collin Thibault*, within the scope of the ```M203 Alternative finance class```

The aim of this project was to develop a model capable of predicting professionnal basketball player injuries, based on box score individual statistics from their latest games, with the final aim of making financial gains from an enhanced betting strategy. *(last update: April 18th)*


*Data extraction*: we manually coded an extraction algorithm for all required statistics, alongside cleaning and enhancing segments to make the dataset usable.

```python
df_stats.sample(10)
```

*Neural nets training*: we implemented an artificial neural network using focal loss and various features engineering and selection techniques.

```python
model.compile(optimizer=optimizer, loss=focal_loss(gamma=2.0, alpha=0.25), metrics=[AUC()])
```

*Investment strategy*: we leveraged the use of deep learning to prompt an injury watch list and make financial profit off betting against injury-prone players with critical influence.

```python
injury_watch_sorted.head(5)
```

All code is ours except when stated otherwise. Thanks for this semester!
