import subprocess
from pathlib import Path

from utils.paths import GGUF_SCRIPT

def convert_to_gguf(merged_path: str, gguf_output: str, outtype: str = "q4_k_m"):
    print(f"Converting model from {merged_path} to GGUF format and saving to {gguf_output}...")

    Path(gguf_output).parent.mkdir(parents=True, exist_ok=True)

    command = [
        "python",
        str(GGUF_SCRIPT),
        merged_path,
        "--outfile",
        gguf_output,
        "--outtype",
        outtype
    ]

    subprocess.run(command, check=True)

    print(f"Model converted to GGUF format and saved to {gguf_output}")