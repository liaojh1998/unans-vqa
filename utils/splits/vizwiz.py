import argparse
import json
import os
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True,
                        help="annotation directory that contains train.json")
    parser.add_argument("--num", type=int, default=2000,
                        help="number of examples to be placed in "
                             "the validation set")
    args = parser.parse_args()

    with open(os.path.join(args.dir, "train.json"), "r") as f:
        jsonl = json.load(f)
        random.shuffle(jsonl)

        # Train
        train = jsonl[:-args.num]
        with open(os.path.join(args.dir, "train_new.json"), "w") as fw:
            json.dump(train, fw)

        # Val
        val = jsonl[-args.num:]
        with open(os.path.join(args.dir, "train_val.json"), "w") as fw:
            json.dump(val, fw)
