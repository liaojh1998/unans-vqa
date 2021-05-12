import argparse
import json
import os
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True,
                        help="json file of annotations")
    parser.add_argument("--out", required=True,
                        help="output json file name")
    args = parser.parse_args()

    with open(args.ann, "r") as f:
        jsonl = json.load(f)
        nums = len(jsonl['questions'])
        swaps = nums // 4
        count = 0

        for i in range(nums):
            jsonl['questions'][i]['answerable'] = 1

        while count < swaps:
            a, b = random.randrange(nums), random.randrange(nums)
            if a != b and jsonl['questions'][a]['image_id'] != jsonl['questions'][b]['image_id']:
                tmp = jsonl['questions'][a]['image_id']
                jsonl['questions'][a]['image_id'] = jsonl['questions'][b]['image_id']
                jsonl['questions'][b]['image_id'] = tmp
                jsonl['questions'][a]['answerable'] = 0
                jsonl['questions'][b]['answerable'] = 0
                count += 1

        with open(args.out, "w") as fw:
            json.dump(jsonl, fw)
