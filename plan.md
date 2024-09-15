# Fantasy hockey bayesian modelling

## GOAL: 
- Fantasy points prediction as a random distribution, for each player. 
- Want to be able to compute summary statistics like average, variance, credible intervals.
- Want to be able to visualize the distribution of different players for comparisons.
- Want to be able to perform comparisons like P(player_1 > player_2)

- Output players that have been missed/not been modelled when compared to a list of players available for drafting.

## DATA:
### Have
- kkupfl scoring working sheet: historical fantasy points for each player (going back to 2017-18 season)
  - team
  - position
  - games played
  - goals
  - assists
  - points
  - shots on goal
  - power play goals
  - power play assists
  - short handed goals
  - short handed assists
  - hits
  - blocked shots 
  - wins
  - goals against
  - saves
  - shutouts
## Preprocessing
- Add a column for fantasy points
- Remove goalies
- Remove players with less than 20 games played
- Remove the Sebastian Aho defenseman
- Standardize data
- Keep only players that are there for all years

### Want
- Age
- Seasons of experience
- Draft position
- Salary
- Time on ice
- Power play time on ice
- Short handed time on ice
- Faceoff wins
- Faceoff losses

## Models
- Bayesian Network
  - Prior: Normal distribution
  - Likelihood: Poisson distribution
  - Posterior: Normal distribution

## Evaluation
- Cross validation
  - Train on 2017-18, 2018-19, evaluate on 2019-20
  - Train on 2020-21, 2021-22, evaluate on 2022-23
  - Train on 2022-23, 2023-24 for final model

## Other Ideas
- Neural Networks
- First perform clustering, then model each cluster separately