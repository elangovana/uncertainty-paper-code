from scipy.stats import binom


def probability_between_heads(m, n, N, p=0.5):
    """
    Calculates the probability of getting between m and n heads (inclusive)
    in N coin tosses, with a probability of heads p.

    Args:
        m (int): Minimum number of heads (inclusive).
        n (int): Maximum number of heads (inclusive).
        N (int): Total number of coin tosses.
        p (float): Probability of getting heads in a single toss (default is 0.5 for a fair coin).

    Returns:
        float: The probability of the number of heads being between m and n.
    """
    # Ensure m is less than or equal to n
    if m > n:
        m, n = n, m

    # The CDF gives P(X <= k).
    # P(m <= X <= n) = P(X <= n) - P(X < m) = P(X <= n) - P(X <= m-1)
    prob_at_most_n = binom.cdf(n, N, p)
    prob_at_most_m_minus_1 = binom.cdf(m - 1, N, p)

    probability = prob_at_most_n - prob_at_most_m_minus_1
    return probability


# --- Example Usage ---
# Calculate the probability of getting between 4 and 6 heads in 10 coin tosses (fair coin)

total_tosses = 100
prob_heads = 0.6  # For an "omni" (fair) coin

min_heads = 50
max_heads = 70
result = probability_between_heads(min_heads, max_heads, total_tosses, prob_heads)

print(f"The probability of getting between {min_heads} and {max_heads} heads in {total_tosses} tosses is: {result:.4f}")

min_heads = 80
max_heads = 100
result = probability_between_heads(min_heads, max_heads, total_tosses, prob_heads)

print(f"The probability of getting between {min_heads} and {max_heads} heads in {total_tosses} tosses is: {result:.4f}")

