from parser.parser import EvalArgParser

def main(args):
    print(args)

if __name__ == '__main__':
    eval_parser = EvalArgParser()
    main(eval_parser.get_arguments())