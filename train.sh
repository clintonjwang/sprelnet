#! /bin/bash
#only run slurm commands if directiory does not already exist 
if [ ! -d /data/vision/polina/users/clintonw/code/placenta/placenta/slurm-output/$1 ]
then
    mkdir /data/vision/polina/users/clintonw/code/placenta/placenta/slurm-output/$1
    sbatch <<EOT
#! /bin/bash
#SBATCH --partition=gpu --qos=gpu
#SBATCH --time=300:00:00
#SBATCH --mem=10G
#SBATCH -J $1
#SBATCH --gres=gpu:1
#SBATCH -e /data/vision/polina/users/clintonw/code/placenta/placenta/slurm-output/$1/logs.err
#SBATCH -o /data/vision/polina/users/clintonw/code/placenta/placenta/slurm-output/$1/logs.out
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --exclude=rosemary,perilla,peppermint,bergamot,curcum,sassafras,clove,lemongrass,anise,mint,caraway,aniseed,turmeric

cd /data/vision/polina/users/clintonw/code/placenta/placenta
source .bashrc
python train_placenta_refactor.py --job_id $1 --config_path ${2:-/data/vision/polina/users/clintonw/code/placenta/train_placenta_config.yaml}
exit()
EOT
else
    echo argument error: id already exists, no job submitted #arbitrary output 
fi

#
#olida,sassafras,jimbu
# --exclusive