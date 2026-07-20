import argparse
import subprocess


cmd_parser = argparse.ArgumentParser()
cmd_parser.add_argument("--model_name", required=True)

arg = cmd_parser.parse_args()

download_cmd = ["hf", "download", arg.model_name]

try:
    subprocess.run(download_cmd, check=True)
    print(f"Model downloaded")
except subprocess.CalledProcessError as e:
    print(f"An error occured while downloading:\n\n{e}")
except FileNotFoundError:
    print(f"hf command not found")