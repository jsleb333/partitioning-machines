

def wedderburn_etherington(n):
    """
    Computes the Wedderburn Etherington numbers. They corresponds to the number of structurally non-equivalent binary trees with 'n' leaves.
    Args:
        n (int): The number of leaves of the binary tree.
    """
    if n == 0:
        return 0
    if n == 1:
        return 1

    if n % 2 == 1:
        val = 0
        for i in range(1, n//2+1):
            val += wedderburn_etherington(i) * wedderburn_etherington(n-i)
        return val

    if n % 2 == 0:
        val = 0
        for i in range(1, n//2):
            val += wedderburn_etherington(i) * wedderburn_etherington(n-i)
        val += wedderburn_etherington(n//2)*(wedderburn_etherington(n//2) + 1)//2
        return val
