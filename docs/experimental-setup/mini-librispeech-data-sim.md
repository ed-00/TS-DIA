experimental setup

## Data Simulation

Following the experimental design, we simulate mixtures with varying numbers of speakers and β parameters:

| Dataset | Split | #Spk | #Mixtures | β |
|---------|-------|------|-----------|---|
| Sim1spk | Train | 1    | 100,000   | 2 |
|         | Val   | 1    | 10,000    | 2 |
|         | Test  | 1    | 100,000   | 2 |
| Sim2spk | Train | 2    | 100,000   | 2 |
|         | Val   | 2    | 10,000    | 2 |
|         | Test  | 2    | 500       | 2 |
|         | Test  | 2    | 500       | 3 |
|         | Test  | 2    | 500       | 5 |
| Sim3spk | Train | 3    | 100,000   | 5 |
|         | Val   | 3    | 10,000    | 5 |
|         | Test  | 3    | 500       | 5 |
|         | Test  | 3    | 500       | 7 |
|         | Test  | 3    | 500       | 11|
| Sim4spk | Train | 4    | 100,000   | 9 |
|         | Val   | 4    | 10,000    | 9 |
|         | Test  | 4    | 500       | 9 |
| Sim5spk | Train | 5    | 100,000   | 13|
|         | Val   | 5    | 10,000    | 13|
|         | Test  | 5    | 500       | 13|

# pretraining 
The pretraining process will be conducted on a combitnation of all the simulated mixures to train for 100 epochs. 

25 epochs on data types

- Meeting: 
    - ami 
    - icsi
    - aishell 4 

- In the wild
    - AVA_avd
    - voxconverse 
    - MsWild  
    - ego-4d


