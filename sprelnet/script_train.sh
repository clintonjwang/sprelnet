#!/bin/bash
#only run slurm commands if directiory does not already exist 
JOBNAME=$1
while [ -d "/data/vision/polina/users/clintonw/code/sprelnet/job_outputs/${JOBNAME}" ]
do
  JOBNAME="${JOBNAME}_"
done
mkdir "/data/vision/polina/users/clintonw/code/sprelnet/job_outputs/${JOBNAME}"
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu --qos=gpu
#SBATCH --time=300:00:00
#SBATCH --mem=10G
#SBATCH -J $1
#SBATCH --gres=gpu:1
#SBATCH -e /data/vision/polina/users/clintonw/code/sprelnet/job_outputs/$1/logs.err
#SBATCH -o /data/vision/polina/users/clintonw/code/sprelnet/job_outputs/$1/logs.out
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

cd /data/vision/polina/users/clintonw/code/sprelnet/sprelnet
PATH=/data/vision/polina/users/clintonw/bin:/data/vision/polina/users/clintonw/anaconda3/bin:/data/vision/polina/shared_software/fsl/bin:/usr/local/csail/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source activate cuda11
python train.py --job_id $JOBNAME --config_path ${2:-/data/vision/polina/users/clintonw/code/sprelnet/configs/$1.yaml}
exit()
EOT

# --exclude=rosemary,perilla,peppermint,bergamot,curcum,sassafras,clove,lemongrass,anise,mint,caraway,aniseed,turmeric
#olida,sassafras,jimbu
# --exclusive
# source .bashrc