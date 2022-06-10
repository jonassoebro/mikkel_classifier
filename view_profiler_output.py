import pstats
p = pstats.Stats('make_dataset_profile')
p.sort_stats('cumulative').print_stats(10)