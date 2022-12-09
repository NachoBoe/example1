import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str, required=False,help="path to output trained model")
ap.add_argument("-p", "--ploter", type=str, required=True,help="path to output loss/accuracy plot")

args = vars(ap.parse_args())
print(args)