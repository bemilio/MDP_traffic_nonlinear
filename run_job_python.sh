# Request one node with n free processor cores
#PBS -l nodes=1:ppn=4:typel
# Mail me when the job ends for any reason
#PBS -m ae
#PBS -M e.benenati@tudelft.nl

# Go to the directory where I entered the qsub command
cd $PBS_O_WORKDIR

# Run
rm -f log.txt
source mdp_env/bin/activate
python3 main.py > log.txt
