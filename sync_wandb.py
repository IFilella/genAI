import os
import subprocess
from datetime import datetime, timedelta
from glob import glob
import wandb

# Path to store the last sync time
last_sync_file = 'last_sync.txt'

# Function to get the last sync time
def get_last_sync_time():
    if os.path.exists(last_sync_file):
        with open(last_sync_file, 'r') as file:
            last_sync_time = datetime.fromisoformat(file.read().strip())
    else:
        # If no sync has been done, default to a time far in the past
        last_sync_time = datetime.now() - timedelta(days=365)
    return last_sync_time

# Function to update the last sync time
def update_last_sync_time():
    with open(last_sync_file, 'w') as file:
        file.write(datetime.now().isoformat())

# Function to list local runs
def list_local_runs():
    local_runs = [path for path in glob("wandb/offline-run-*")]
    return local_runs

def get_update_time(run):
    
    files = glob(run + "/**", recursive=True)
    
    latest_mod_time = max([os.path.getmtime(run)] + [os.path.getmtime(f) for f in files])
    return datetime.fromtimestamp(latest_mod_time)

# Function to filter updated runs
def filter_updated_runs(runs, last_sync_time):
    
    updated_runs = []
    for run in runs:
    
        run_update_time = get_update_time(run)
        
        # print(run, run_update_time, last_sync_time)
        
        if run_update_time > last_sync_time:
            updated_runs.append(run)
    
    return updated_runs

import subprocess

def sync_runs(run_ids):
    for run_id in run_ids:
        print(os.getcwd())
        print(run_id)
        process = subprocess.Popen(
            ["wandb", "sync", run_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            while True:
                # Read line from stdout
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    if "ERROR" in output:
                        print(f"Run {run_id} sync encountered an ERROR and was aborted.")
                        process.terminate()
                        break

                # Read line from stderr
                error = process.stderr.readline()
                if error == '' and process.poll() is not None:
                    break
                if error:
                    print(error.strip())
                    if "ERROR" in error:
                        print(f"Run {run_id} sync encountered an ERROR and was aborted.")
                        process.terminate()
                        break

            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"Run {run_id} sync timed out.")
            process.terminate()
        except Exception as e:
            print(f"An error occurred: {e}")
            process.terminate()


def get_online_runs():
    
    api = wandb.Api()
    runs = api.runs("eapm-bsc/train_decoder_hyperparameters")
    
    run_ids = [run.id for run in runs]
    
    return run_ids  

# Main script
if __name__ == "__main__":

    last_sync_time = get_last_sync_time()
    print(f'Last sync time: {last_sync_time}')
    local_runs = list_local_runs()
    print(local_runs)
    online_runs = get_online_runs()
    
    # local_runs = [run_path for run_path in local_runs if run_path.split("-")[-1] in online_runs]
    # print(local_runs)
    
    updated_runs = filter_updated_runs(local_runs, last_sync_time)
    print(updated_runs)

    if updated_runs:
        print(f'Syncing {len(updated_runs)} updated runs...')
        sync_runs(updated_runs)
        update_last_sync_time()
    else:
        print('No runs to sync.')

