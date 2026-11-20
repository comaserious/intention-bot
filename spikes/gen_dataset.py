# One-off: generate eval/dataset.toml from the spike's GIVEN/EXTRA lists.
import re

src = open("spikes/spike_ab2.py", encoding="utf-8").read()


def grab(name):
    return eval(re.search(name + r" = (\[.*?\n\])", src, re.S).group(1))


def q(s):
    return '"' + s.replace(chr(92), chr(92) * 2).replace('"', chr(92) + '"') + '"'


out = []
for u, e in grab("GIVEN") + grab("EXTRA"):
    out.append("[[case]]\nutterance = " + q(u) + "\nexpected = [" + ", ".join(q(x) for x in sorted(e)) + "]\n")
open("eval/dataset.toml", "w", encoding="utf-8").write("\n".join(out))
