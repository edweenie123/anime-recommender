# Anime Recommender

An autoencoder based recommender system trained to curate anime recommendations for
MyAnimeList users.
Users may enter their MAL usernames into a website, with
a VueJS frontend and Flask backend,
that processes their anime lists (obtained via the MAL API) 
to generate anime recommendations.

### Demo 

![demo](gif/demo.gif)

### The Model

The architecture of the recommender system model was inspired by 
the [Autorec paper](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf) 
which uses autoencoders to perform collaboriative filtering.
Training was done using PyTorch on Google Colaboratory's hardware. 


The dataset for the model was obtained by aggregating, cleaning and preprecessing
the following Kaggle datasets

- [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020?select=rating_complete.csv)
- [MyAnimeList Anime and Manga Datasets](https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist)

using Pandas.


### Installation

To run the project locally, npm 8.18.0 
and conda 22.9.0 are required before performing 
the following steps.


1. Clone the application

```bash
git clone https://github.com/edweenie123/anime-recommender.git
cd anime-recommender
```

2. Install required Python packages onto a conda environment

```bash
conda create --name <env> --file requirements.txt
```

3. Run the Flask server by `cd`-ing into `web/server`  and running

```bash
flask run
```
> NB: Make sure to create a `config.py` file in `web/server` containing your MAL API access token

4. Run the VueJS development server by `cd`-ing into `web/client`  and running

```bash
npm run dev
```

