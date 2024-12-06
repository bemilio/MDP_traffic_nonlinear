This repository contains the python implementation accompanying the paper:  

**Probabilistic game-theoretic traffic routing**  
*Emilio Benenati, Sergio Grammatico*  
*IEEE Transactions on intelligent transportation systems, 2024*  
DOI: [10.1109/TITS.2024.3399112](https://doi.org/10.1109/TITS.2024.3399112)  

## Description  
This code solves for the Nash equilibrium routing solution for a fleet of vehicles. The solution can either be computed offline, or offline via a receding-horizon controller.

## Dependencies  
See `requirements.txt`

## Running the code
To execute the code:

- Create the test graph
```
python create_test_graph.py
```
- Compute offline routing solution
```
python main.py
```
- Store the resulting saved file in a folder and update [this line](https://github.com/bemilio/MDP_traffic_nonlinear/blob/6b4e42e28fe73048ae0431b096b808a14e5eecd2/plot.py#L30) accordingly
- Plot offline routing solution
```
python plot.py
```
- Compute receding-horizon routing solution
```
python main_multiperiod.py
```
- Store the resulting saved file in a folder and update [this line](https://github.com/bemilio/MDP_traffic_nonlinear/blob/6b4e42e28fe73048ae0431b096b808a14e5eecd2/plot_multiperiod.py#L30) accordingly
- Plot results of the receding horizon solution
```
python plot_multiperiod.py
```


In the simulation section of the referenced paper paper, the results are obtained by running the code multiple times with randomized parameters and initial condition (see paper for details). The generated data used in the paper is available at [this link](...)
