import tensorflow as tf
from itertools import combinations_with_replacement

class MultivariateCandidateFunctionConstructor:
    """Construct multivariate candidate functions by multiplying univariate candidates."""

    def __init__(self, multiplicative_order, include_constant=True, data_type='float32'):
        """Initialize the constructor.

        Args:
            multiplicative_order: Maximum order of multiplication
            include_constant: Whether to include a constant term
            data_type: Data type (default: 'float32')
        """
        self.multiplicative_order = multiplicative_order
        self.include_constant = include_constant
        self.data_type = data_type

    def construct(self, univariate_candidates):
        """Construct multivariate candidate functions.

        Args:
            univariate_candidates: Tensor of shape (N, D) where each column
                                  is a univariate candidate function

        Returns:
            Tensor of shape (N, M) with all unique multiplicative combinations
        """
        data = tf.cast(univariate_candidates, self.data_type)
        n_samples = tf.shape(data)[0]
        n_univariate = data.shape[1]

        candidates = []

        # Add constant term if requested
        if self.include_constant:
            candidates.append(tf.ones((n_samples, 1), dtype=self.data_type))

        # Generate unique combinations for each order
        for order in range(1, self.multiplicative_order + 1):
            for combination in combinations_with_replacement(range(n_univariate), order):
                term = tf.ones((n_samples, 1), dtype=self.data_type)
                for col_idx in combination:
                    term = term * data[:, col_idx:col_idx+1]
                candidates.append(term)

        return tf.concat(candidates, axis=1)


if __name__ == "__main__":
    # Example
    x = tf.constant([[1.0, 2.0],
                     [3.0, 4.0],
                     [5.0, 6.0]], dtype='float32')

    constructor = MultivariateCandidateFunctionConstructor(multiplicative_order=2)
    result = constructor.construct(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result.shape}")
    print(f"\nOutput:\n{result.numpy()}")
