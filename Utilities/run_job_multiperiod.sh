# Request one node with n free processor cores
#PBS -l nodes=1:ppn=4:typel
# Mail me when the job ends for any reason
#PBS -m ae
#PBS -M e.benenati@tudelft.nl
#PBS -N MDP_multiperiod

# Go to the directory where I entered the qsub command
cd $PBS_O_WORKDIR

# Run
source ~/venv_MDP/bin/activate
python main_multiperiod.py
