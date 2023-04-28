import pandas


def read(labels):
    data = pandas.read_csv(labels)
    print("data shape ", data.shape, type(data))
