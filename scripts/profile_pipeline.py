import cProfile
import pstats
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.run_metabonet_hybrid_streaming import main

def profile():
    sys.argv = ['run_metabonet_hybrid_streaming.py', '--stage', 'smoke']
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(50)
    print(s.getvalue())

if __name__ == '__main__':
    profile()
