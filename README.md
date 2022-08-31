# What Makes a Hit: Predicting Hit Songs with a Multi-Input Neural Network

This independent research project is the first of its kind to experiment with predicting hit songs with a multi-input neural network.
To do this, I crafted the most extensive multi-modal dataset for this task, combining both audio and lyric features.

Data collection was distributed among multiple EC2 instances on AWS and the resulting data was stored in a Postgres database with RDS.

The experiments and their results are in `Spotify_Analysis.ipynb`.
The resulting accuracy of the neural network was `86.4%`, which is the highest accuracy of any NN in the hit prediction domain.
