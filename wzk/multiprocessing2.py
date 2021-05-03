import numpy as np
import multiprocessing as mp
import os


from time import sleep
from wzk import change_tuple_order


# FINDING h5py objects can not be pickled
#   -> read the files inside the function instead / h5py objects are only references anyway
#   Joblib: Threading backend allows shared memory

# FINDING

def n_processes_wrapper(n_processes, n_samples):
    return min(max(1, n_processes), n_samples)


def get_n_samples_per_process(*, n_samples, n_processes):
    n_samples_per_core = n_samples // n_processes
    odd_idx = n_samples - (n_samples_per_core * n_processes)
    n_samples_per_core = np.repeat(n_samples_per_core, repeats=n_processes)
    n_samples_per_core[:odd_idx] += 1

    n_samples_per_core_cs = np.cumsum(np.concatenate([[0], n_samples_per_core]))

    return n_samples_per_core, n_samples_per_core_cs


def mp_wrapper(*args, fun, n_processes=1, max_chunk_size=None, __debug_loop=False):
    """
    Multiprocessing Wrapper for a function with a single argument.
    arg must be an iterative and will be split along its first dimension and fed to the different processes
    The objects returned from fun must be numpy arrays and also be splittable / combinable along the first dimension

    Caveats:
    - The function breaks if the data passed through the pipe is to large
      so make sure that the data shape does not exceed ~100Mb per process
      https://stackoverflow.com/questions/31552716/multiprocessing-queue-full

    - Numpy's random number generator starts with the same seed for each process, so if your calculations depend on
      those random values, all processes will work with the same numbers. Default behaviour is to call
      np.random.seed() in each process, to make sure the random numbers of each process are different. If you want
      them to be the same use np.random.seed(42) in the function directly
      https://stackoverflow.com/a/12915206/7570817

    - Be careful with functions that only return a value which shape does not depend on the input,
      In such cases different number of process lead to different results, especially the keyword max_chunk_size
      might cause trouble

    - max_chunk_size is intended for cases where both a loop + parallelization is needed because of memory limitations
    """

    time_sleep = 0.01  # s # TODO smaller time ?
    time_sleep2 = 0.01  # s # TODO smaller time ?

    if len(args) == 0:
        n_samples = n_processes
    elif isinstance(args[0], int):
        n_samples = args[0]
    else:
        n_samples = np.shape(args[0])[0]

    n_processes = n_processes_wrapper(n_processes=n_processes, n_samples=n_samples)

    if n_processes == 1:
        return fun(*args)

    if len(args) == 0:
        def __fun_wrapper(i_process, queue):
            queue.put((i_process, fun()))
    else:
        n_samples_pp, n_samples_pp_cs = \
            get_n_samples_per_process(n_samples=n_samples, n_processes=n_processes)

        if isinstance(args[0], int):
            if max_chunk_size is not None:
                def fun_i(i_process):
                    np.random.seed()
                    n_s = n_samples_pp[i_process]
                    ns_pp, ns_pp_cs = get_n_samples_per_process(n_samples=n_s, n_processes=n_s // max_chunk_size)
                    return combine_results([fun(ns_pp_i) for ns_pp_i in ns_pp])
            else:
                def fun_i(i_process):
                    np.random.seed()
                    return fun(n_samples_pp[i_process])
        else:
            if max_chunk_size is not None:
                def fun_i(i_process):
                    np.random.seed()
                    n_s = n_samples_pp[i_process]
                    ns_pp, ns_pp_cs = get_n_samples_per_process(n_samples=n_s, n_processes=n_s // max_chunk_size)
                    return combine_results([fun(*map(lambda a: a[ns_pp_cs[j]:ns_pp_cs[j+1]], args))
                                            for j in range(len(ns_pp_cs)-1)])
            else:
                def fun_i(i_process):
                    np.random.seed()
                    return fun(*map(lambda a: a[n_samples_pp_cs[i_process]:n_samples_pp_cs[i_process+1]],
                                    args))

        if __debug_loop:
            return combine_results([fun_i(i_process=ip) for ip in range(n_processes)])

        def __fun_wrapper(i_process, queue):
            queue.put((i_process, fun_i(i_process=i_process)))

    # Start the processes and save their results in a queue
    result_queue = mp.Queue(n_processes)
    process_list = []
    for i in range(n_processes):
        p = mp.Process(target=__fun_wrapper, args=(i, result_queue), name=str(i))
        p.start()
        process_list.append(p)

    # Make sure the queue is full before joining the processes with a small timeout, otherwise there occurred
    # errors, because a process filled the queue but was still alive
    while True:
        sleep(time_sleep)
        # print(result_queue.qsize())
        if result_queue.full():
            break

    # Wait for the processes to finish
    for i in range(n_processes):
        process_list[i].join(timeout=time_sleep2)

    # Combine and return the results
    results = [result_queue.get() for _ in range(n_processes)]

    # - order the results according to the process indices
    results = change_tuple_order(results)
    idx = np.argsort(results[0])
    results = [results[1][i] for i in idx]

    return combine_results(results=results)


def combine_results(results):
    if isinstance(results[0], tuple):
        results = change_tuple_order(results)
        return tuple([np.concatenate(r, axis=0) if np.ndim(r[0]) > 0 else np.array(r)
                      for r in results])
    else:
        if results[0] is None:
            return None
        else:
            return np.concatenate(results, axis=0)


def test_mp_wrapper():
    from wzk.dicts_lists_tuples import list_allclose

    def fun__int_in(n):
        return np.zeros((n, 3))

    def fun__arr_in(arr):
        arr[:] = 0
        return arr

    def fun__multiple_arr_in(a1, a2, a3):
        a4 = a1 + a2 + a3
        return a4

    def fun__multiple_arr_out(n):
        return tuple(np.full((n, ii), ii) for ii in range(1, 5))

    def fun__multiple_out(n):
        return tuple(np.full((n, ii), ii) for ii in range(1, 3)) + (11, 12)

    n_processes = 10
    z = np.zeros((1000, 3))
    a = np.ones((1000, 3))
    b = np.ones((1000, 3)) * 2
    c = np.ones((1000, 3)) * 3

    res1 = mp_wrapper(999, fun=fun__int_in, n_processes=n_processes)
    res1b = mp_wrapper(999, fun=fun__int_in, n_processes=n_processes, __debug_loop=True)
    assert np.allclose(res1, np.zeros((999, 3)))
    assert list_allclose(res1, res1b)

    res2 = mp_wrapper(a, fun=fun__arr_in, n_processes=n_processes)
    res2b = mp_wrapper(a, fun=fun__arr_in, n_processes=n_processes, __debug_loop=True)
    assert np.allclose(res2, z)
    assert list_allclose(res2, res2b)

    res3 = mp_wrapper(a, b, c, fun=fun__multiple_arr_in, n_processes=n_processes)
    res3b = mp_wrapper(a, b, c, fun=fun__multiple_arr_in, n_processes=n_processes, __debug_loop=True)
    assert np.allclose(res3, a+b+c)
    assert list_allclose(res3, res3b)

    res4 = mp_wrapper(989, fun=fun__multiple_arr_out, n_processes=n_processes)
    res4b = mp_wrapper(989, fun=fun__multiple_arr_out, n_processes=n_processes, __debug_loop=True)
    for i in range(1, 5):
        assert np.allclose(res4[i-1], np.full((989, i), i))
    assert all(list_allclose(res4, res4b))

    res5 = mp_wrapper(1007, fun=fun__multiple_out, n_processes=n_processes)
    res5b = mp_wrapper(1007, fun=fun__multiple_out, n_processes=n_processes, __debug_loop=True)
    for i in range(1, 3):
        assert np.allclose(res5[i-1], np.full((1007, i), i))
    assert all(list_allclose(res5, res5b))

    res6 = mp_wrapper(1007, fun=fun__multiple_out, n_processes=n_processes, max_chunk_size=10)
    res6b = mp_wrapper(1007, fun=fun__multiple_out, n_processes=n_processes, __debug_loop=True)
    for i in range(1, 3):
        assert np.allclose(res6[i-1], np.full((1007, i), i))
    assert all(list_allclose(res6[:2], res6b[:2]))


if __name__ == '__main__':
    test_mp_wrapper()
