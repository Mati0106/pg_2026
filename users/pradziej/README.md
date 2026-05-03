# Prereqisites

  ## Install depenencies and enable the virtual env.

  ```bash
    conda env create --file=users/pradziej/environments.yml
  ```

  ## Activate the env

  ```bash
    conda activate pr_pg26
  ```

  ## Prerequisites 

  If the data set isn't cached (in default datasets directory) you have to download it either from kaggle or manually (what is simpler). 
  For direct kaggle download put the API key either:
  
  - to env variable 
  ```bash
    export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx
  ```

  - to file `~/.kaggle/access_token`
  - it will ask you for the token...

  ## Run

  ```bash
    python3 users/pradziej/main.py
  ```
  
## About the dataset

It seams to be a bit ridicolous that we have few girls in age around 10 that are in pregnat (but it's possible, brutal but possible).
https://www.who.int/news-room/fact-sheets/detail/adolescent-pregnancy
https://data.unicef.org/topic/child-health/early-childbearing/

Ont the opposite there is a woman that is over 70 years old and also in pregnat, tchnically it's possible:
https://en.wikipedia.org/wiki/Erramatti_Mangamma

The whole dataset comes from India and from IoT sensors -> TBH. I don't beleve that it is real or enough complete and fine that could be used in real life. 
So models in this projects probably are not adapable in real life.

