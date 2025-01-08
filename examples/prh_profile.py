"""
Module to run profile on like so:

rm -f program.prof; python prh_profile.py ; snakeviz program.prof
"""
import sys
import cProfile
import pstats
sys.path.append("../")
import mol_cv
import load_medchem_data
import predict_medchem

def run_many_times(repeats,data,predictor_dict):
    """
    Fit the data multiple times

    :param repeats: how many times to repeat
    :param data: list length N of mols
    :param predictor_dict: see calculate_properties
    :return: nothing, just runs a bunch
    """
    for _ in range(repeats):
        mol_cv.calculate_properties(data,predictor_dict)

def run():
    """
    Runs the profiling code
    """
    fda = load_medchem_data.load_fda_drugs()
    predictor_dict = predict_medchem.all_predictors()
    profiler = cProfile.Profile()
    profiler.enable()
    run_many_times(repeats=3,data=fda["mol"],predictor_dict=predictor_dict)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('program.prof')


if __name__ == "__main__":
    run()
