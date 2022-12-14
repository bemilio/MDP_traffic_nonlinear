# Probabilistic game-theoretic traffic routing
This is the code associated to the paper "Probabilistic game-theoretic traffic routing" - E. Benenati and S. Grammatico, 2022 (under revision).

This code solves for the Nash equilibrium routing solution for a fleet of vehicles. To run the code:

- Create the test graph
```
python create_test_graph.py
```
- Pre-compute complete routing solution
```
python main.py
```
- Plot results
```
python plot.py
```
- Receding-horizon routing solution
```
python main_multiperiod.py
```
- Plot results of the receding horizon solution
```
python plot_multiperiod.py
```
