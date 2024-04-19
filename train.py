from parser.parser import TrainArgParser

def main(args):
    print(args)

if __name__ == '__main__':
    train_parser = TrainArgParser()
    main(train_parser.get_arguments())