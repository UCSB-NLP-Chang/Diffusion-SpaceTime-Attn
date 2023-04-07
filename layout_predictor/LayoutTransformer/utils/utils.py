import os
import json
import re

def ensure_dir(dir):
    os.makedirs(dir, exist_ok=True)
