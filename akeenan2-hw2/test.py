import decode
import train
import argparse
import cer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store_true', help="decoder")
    parser.add_argument('-t', action='store_true', help="trainer")
    parser.add_argument('-f', nargs=1, help="output lines to file, ex. data/tmp.new")
    args = parser.parse_args()

    if args.d:
        decoder = decode.FSTM_Decode()
        decoder.train()
        if args.f: # if file specified, run on whole file
            decoder.test_all(args.f[0])
        else: # otherwise, just run on the first 10 lines
            decoder.test()
    # needs a file to run
    elif args.t and args.f:
        trainer = train.FSTM_Train()
        trainer.test(args.f[0])
