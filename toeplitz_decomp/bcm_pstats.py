#!/usr/bin/env python

import pstats
p = pstats.Stats('64_128.out')
p.strip_dirs().sort_stats('cumulative').print_stats(20)

p1 = pstats.Stats('64_512.out')
p1.strip_dirs().sort_stats('cumulative').print_stats(25)
