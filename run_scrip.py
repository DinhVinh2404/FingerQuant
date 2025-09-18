import os
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, choices=['thumb_classifier1'])
    args = parser.parse_args()
    w_bits = [8]
    a_bits = [8]

    if args.exp_name == "thumb_classifier1":
        for i in range(1):
            os.system(f"python main.py --data_path data_custom --arch thumb_classifier1 --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.02")
            time.sleep(0.5)