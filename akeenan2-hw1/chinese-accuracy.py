import argparse
import bigram
import operator

def predict(predict_file, correct_file):
    """Predict the text file on the model."""
    total = 0.
    correct = 0

    with open(predict_file) as predict_f, open(correct_file) as correct_f:
        for predict_line, correct_line in zip(predict_f, correct_f):
            for token, correct_token in zip(predict_line.split(), correct_line):
                predict_token = model.predict(token)
                if predict_token == correct_token:
                    correct += 1
                total += 1
                model.read(correct_token)

    print("Accuracy = " + str(correct/total*100) + "%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--charmap', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--predict', required=True)
    parser.add_argument('--correct', required=True)
    args = parser.parse_args()

    model = bigram.Bigram()
    model.read_charmap(args.charmap)
    model.train(args.train)
    model.start()
    predict(args.predict, args.correct)
