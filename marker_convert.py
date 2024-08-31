import argparse
import subprocess
import pkg_resources


def main():
    parser = argparse.ArgumentParser(description="Convert a folder of PDFs to a folder of markdown files in chunks.")
    parser.add_argument("in_folder",type=str, help="Input folder with pdfs.")
    parser.add_argument("out_folder",type=str, help="Output folder.")
    parser.add_argument("num_device", help="Number of devices.")
    parser.add_argument("num_workers", help="Number of workers on each device.")
    args = parser.parse_args()
    
    for idx in range(args.num_device):
        cmd = [f"CUDA_VISIBLE_DEVICES={idx}", "marker", args.in_folder, args.out_folder, "--num_chunks", args.num_device, "--chunk_idx", f"{idx}", "--workers", f"{args.num_workers}"]
        subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    main()