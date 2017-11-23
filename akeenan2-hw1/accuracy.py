import argparse
import unigram
import fivegram
import sevengram_smoothed
import operator

unigram_model = unigram.Unigram()
fivegram_model = fivegram.Fivegram()
sevengram_smoothed_model = sevengram_smoothed.Sevengram_Smoothed()

def predict(filename):
    """Predict the text file on the model."""
    total = 0.
    correct = 0
    for line in open(filename):
        model.start()
        for w in line.rstrip('\n'):
            if model.predict() == w:
                correct += 1
            model.read(w)
            total += 1
    print("Accuracy = " + str(correct/total*100) + "%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--train', required=True)
    parser.add_argument('--predict', required=True)
    args = parser.parse_args()

    if args.model == 'unigram':
        model = unigram_model
    elif args.model == 'fivegram':
        model = fivegram_model
    else:
        model = sevengram_smoothed_model

    model.train(args.train)
    model.start()
    predict(args.predict)
