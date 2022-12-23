"""
    import time
    import numpy as np
    from pybrain.optimization.distributionbased.nes import OriginalNES as ONES


    def ellipsoid(x):
        x = np.power(x, 2)
        return np.dot(np.power(10, 6 * np.linspace(0, 1, x.size)), x)


    solver = ONES(ellipsoid, 4.0*np.ones((2,)), minimize=True, maxEvaluations=2e6, verbose=True, importanceMixing=False)
    start_time = time.time()
    solver.learn()
    print("Runtime: {:7.5e}".format(time.time() - start_time))
    # Numerical Instability. Stopping.  # 16000016
    # Evals: 0 Step: 0 best: None
    # (array([4., 4.]), None)


    def ellipsoid(x):
        x = np.power(x, 2)
        return -np.dot(np.power(10, 6 * np.linspace(0, 1, x.size)), x)


    solver = ONES(ellipsoid, 4.0*np.ones((2,)), maxEvaluations=2e6, verbose=True, importanceMixing=False)
    start_time = time.time()
    solver.learn()
    print("Runtime: {:7.5e}".format(time.time() - start_time))
    # Numerical Instability. Stopping.
    # Evals: 0 Step: 0 best: None
    # (array([4., 4.]), None)
"""
