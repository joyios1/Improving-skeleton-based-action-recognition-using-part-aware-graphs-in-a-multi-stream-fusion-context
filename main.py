#!/usr/bin/env python

from SourceCode.ArgumentParser import get_parser, load_arguments_from_yaml, tab_printer
from SourceCode.Processor import Processor

if __name__ == '__main__':
    # get the parser
    parser = get_parser()

    # load arguments form config ('.yaml') file and parse them
    args = load_arguments_from_yaml(parser)

    # print the parsed arguments
    tab_printer(args)

    # initialize the Processor and start the training
    processor = Processor(args)
    processor.start()
