import multiprocessing as mp


def get_number_of_jobs():
    return int(mp.cpu_count())


def apply_async(array, func):
    pool = mp.Pool(get_number_of_jobs())
    result = pool.map(func, array)
    pool.close()
    return result
