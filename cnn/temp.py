def argHandler(argv):

    import argparse

    parser = argparse.ArgumentParser(description='Test argument parser')

    parser.add_argument("-d", "--dataset", required=True, help="path to input dataset")

    args = parser.parse_args(argv)

    return args.foo, args.bar, args.nee

if __name__=='__main__':

    argList = ["./animals"]

    print(argHandler(argList))
