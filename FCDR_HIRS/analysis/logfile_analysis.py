"""Analyse logfiles coming out of FCDR generation

Make sure current working directory is in 'generate_fcdr'
"""

import subprocess
import pathlib
import itertools
import pprint

cmd = """tail -qn1 $(grep -L "Successfully" */*.out | sed -e 's/out/err/g')  | sort | uniq -c | sort -n"""

# FIXME: read those from common source with code generating them

error_modes = dict(
    granule_contained = "the granule is entirely contained",
    bad_timespan = "Time span appears",
    bad_latspan = "Range of latitude",
    bad_duplicate_removal = "probably means duplicate removal",
    no_header ="Could not find header",
    non_integer_records = "truncated",
    too_few_scanlines = "File contains only",
    too_few_scanlines_2 = "scanlines in period",
    firstline_unable = "Unable to filter firstline",
    incomplete_records = "but I found only",
    unsorted_time = "returned data with unsorted time")

warning_modes = dict(
    out_of_order = "scanlines are out of order, resorting",
    duplicate_scanlines = "duplicate scanlines (judging from scanline number)",
    unflagged_bt0 = "where my T_b estimate is 0 but unflagged",
    manual_nonzero = "non-zero values for manual coefficient",
    reduced_context = "Reduced context available",
    repeated_time = "There are scanlines with different scanline numbers but the same time",
    insufficient_context = "the context does not sufficiently cover the period",
    no_directories = "Found no directories.",
    no_files = "Directories searched appear to contain no matching files.",
    Lself_suspect = "When trying to fit or test self-emission model",
    calibzero = "Found cases where counts_space == counts_iwct == 0.",
    remaining_timeseq = "Still has time sequence issues")

# alternative: collections.Counter
error_counts = dict.fromkeys(error_modes, 0)
warning_counts = dict.fromkeys(warning_modes, 0)

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
            if "ERROR" in line and "Can not read file" in line:
                for k in error_modes.keys():
                    if error_modes[k] in line:
                        error_counts[k] += 1
                        break
                else:
                    unknown.append(line)
            elif "WARNING" in line:
                for k in warning_modes.keys():
                    if warning_modes[k] in line:
                        warning_counts[k] += 1
                        break
                else:
                    unknown.append(line)
                

    if len(unknown) > 0:
        print("Found {:d} unknown errors:".format(len(unknown)))
        print("\n".join(unknown))

    for (k, v) in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:>25s}: {v:<6d} {error_modes[k]:<30s}")
   
    for (k, v) in sorted(warning_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:>25s}: {v:<6d} {warning_modes[k]:<30s}")
    
    for (k, v) in error_modes.items():
        if error_counts[k] > 1000:
            print(f"Worst offenders for {k:s}")
            sts = subprocess.run(
                f"""grep -c "{v:s}" */*.log | sort -t':' -k2 -n | tail""",
                shell=True) 
    for (k, v) in warning_modes.items():
        if warning_counts[k] > 1000:
            print(f"Worst offenders for {k:s}")
            sts = subprocess.run(
                f"""grep -c "{v:s}" */*.log | sort -t':' -k2 -n | tail""",
                shell=True)
