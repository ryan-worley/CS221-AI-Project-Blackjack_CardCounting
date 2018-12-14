Main Scripts for Running: Can call any from command window without input arguments. Must change configuration of main python function
to adjust the settings. Sorry not user friendly yet

BlackJackCardCount - Main script to run value iteration on BlackJackMDP. Uses ValueIteration.py as the algorithm. As currently
configured, will loop through mutliple count and midcard values to solve the MDP, then will output utility and policy
to the policy folder in the project. All policies are already created and stored in the policy folder for use in other scripts.

Simulate - Main simulation script to run simulations of Completed_FullBlackJack_Exact using the policy
learned from BlackJack MDP. Can toggle on and off running simulation analysis. If simulation analysis data
is already collected, there is additional functionality to load data back in and do analysis. 

PredictedBettingStrategyValues - Matlab script that does probability data outputted from simulate.py, specifically
with fitting a probability distribution to count distribution data. Run simulate.py to get count distribution data,
then subsequently run matlab file and it should do everything for you. Must make sure betting schemes are smae in simulate.py
and matlab script as well for accurate results.

Application GUI-GUI application that reads out policy and expected value based off state. 

%------------------------------------------------------------------------------------------

Passive Scripts, Data Accessed from Mains:

BlackJackMDP - Approximate MDP made for blackjack value iteration

ValueIteration - Contains algorithms necessary to solve search problems, simulation

MatlabDataFiles- Saved in from python for use in the matlab file

policy - Folder that contains all optimum policy and utility values in .pkl files for the midcard and count,
these files can be loaded in through python for viewing. This files are also the basis for the database of the GUI 
analysis. Has other miscellaneous useful data as well, some of which is incorporated to matlab files

cauchy - Functions used to generate a cauchy distribution as this not in matlab default interface
 

