import os
import glob
import sys

LOG_FILE = "clean_log_internal.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

log("Starting cleanup...")
data_dir = "data"
if os.path.exists(data_dir):
    files = glob.glob(os.path.join(data_dir, "*"))
    log(f"Found {len(files)} files.")
    for f in files:
        try:
            if os.path.isfile(f):
                os.remove(f)
                log(f"Deleted {f}")
            elif os.path.isdir(f):
                 # skip dirs or use shutil
                 pass
        except Exception as e:
            log(f"Error deleting {f}: {e}")
else:
    log("Data dir does not exist.")

log("Done.")
