import argparse

parser=argparse.ArgumentParser()

parser.add_argument("--foo", help="Foo is number, default=100", default=9)
parser.add_argument("--bar", help="Bar is boolean, default=True", default=True)

args=parser.parse_args()

# User typed parameters are always string
foo = args.foo if type(args.foo) == int else int(args.foo)
bar = args.bar if type(args.bar) == bool else args.bar.lower() in ["true", "1", "yes"]

print(f"foo type: ${type(foo)}, value: ${foo}")
print(f"bar type: ${type(bar)}, value: ${bar}")