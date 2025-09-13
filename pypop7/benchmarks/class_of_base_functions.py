class Rastrigin(BaseFunction):
    def __call__(self, x):
        """

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        y : float
            scalar fitness.
        """
        return rastrigin(x)
