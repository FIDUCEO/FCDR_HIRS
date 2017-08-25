"""Analyse logfiles coming out of FCDR generation

Make sure current working directory is in 'generate_fcdr'
"""

import subprocess
import pathlib
import itertools
import pprint

cmd = """tail -qn1 $(grep -L "Successfully" */*.out | sed -e 's/out/err/g')  | sort | uniq -c | sort -n"""

# FIXME: read those from common source with code generating them

modes = dict(
    granule_contained = "the granule is entirely contained",
    bad_timespan = "Time span appears",
    bad_latspan = "Range of latitude",
    bad_duplicate_removal = "probably means duplicate removal",
    no_header ="Could not find header",
    non_integer_records = "truncated",
    too_few_scanlines = "File contains only",
    firstline_unable = "Unable to filter firstline",
    incomplete_records = "but I found only")

# alternative: collections.Counter
counts = dict.fromkeys(modes, 0)

unknown = []

def main():
#    pprint.pprint(modes)
    cwd = pathlib.Path.cwd()
    if not "generate_fcdr" in cwd.parts:
        raise ValueError("You must be in generate_fcdr directory")
    # check crashing errors
    sts = subprocess.run(cmd, shell=True)
    # check "orbit failed" errors
    
    for logfile in cwd.glob("*/*.log"):
        for line in open(logfile, "r", encoding="utf-8"):
            if not ("ERROR" in line and "Can not read file" in line):
                continue
            for k in modes.keys():
                if modes[k] in line:
                    counts[k] += 1
                    break
            else:
                unknown.append(line)

    if len(unknown) > 0:
        print("Found {:d} unknown errors:".format(len(unknown)))
        print("\n".join(unknown))

    for (k, v) in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:>25s}: {v:<6d} {modes[k]:<30s}")
    
    for (k, v) in modes.items():
        if counts[k] > 1000:
            print(f"Worst offenders for {k:s}")
            sts = subprocess.run(
                f"""grep -c "{v:s}" */*.log | sort -t':' -k2 -n | tail""",
                shell=True)
