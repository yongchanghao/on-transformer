import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

if __name__ == "__main__":
    with open(args.input, encoding="utf8") as r, open(args.output, 'w', encoding="utf8") as w:
        for line in r:
            w.write(" ".join(list(reversed(line.strip().split()))) + '\n')

