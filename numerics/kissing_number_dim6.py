from mpmath import mp

mp.dps = 110


def compute():
    """
    Construct the E6 root system as a kissing configuration in R^6.

    The E6 root system has 72 roots, all of norm sqrt(2). When normalized
    to unit vectors, they form a valid kissing configuration (pairwise dot
    products <= 1/2).

    We build the roots from the E6 Cartan matrix:
    1. Compute simple root coordinates via Cholesky decomposition of the
       Cartan matrix (which equals the Gram matrix for simply-laced types).
    2. Generate all positive roots by iterating: for each known root alpha,
       try alpha + alpha_i for each simple root alpha_i, accepting it if the
       result is a root (determined by the Cartan matrix).
    3. Include negatives to get all 72 roots.
    4. Normalize to unit length and verify the kissing constraint.

    Returns the number of points in the configuration (72).
    """
    # E6 Cartan matrix (Bourbaki labeling, node 2 branches off node 4):
    #     1 - 3 - 4 - 5 - 6
    #             |
    #             2
    cartan = [
        [ 2,  0, -1,  0,  0,  0],
        [ 0,  2,  0, -1,  0,  0],
        [-1,  0,  2, -1,  0,  0],
        [ 0, -1, -1,  2, -1,  0],
        [ 0,  0,  0, -1,  2, -1],
        [ 0,  0,  0,  0, -1,  2],
    ]

    # Cholesky decomposition: Cartan = L L^T
    # The rows of L give the simple root coordinates in R^6.
    n = 6
    L = [[mp.mpf('0') for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = mp.fsum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = mp.sqrt(mp.mpf(cartan[i][i]) - s)
            else:
                L[i][j] = (mp.mpf(cartan[i][j]) - s) / L[j][j]

    simple_roots = [list(row) for row in L]

    def dot(a, b):
        return mp.fsum(a[k] * b[k] for k in range(n))

    def add(a, b):
        return [a[k] + b[k] for k in range(n)]

    def neg(a):
        return [-a[k] for k in range(n)]

    def norm_sq(a):
        return dot(a, a)

    # All roots in E6 have the same norm squared = 2
    root_norm_sq = mp.mpf('2')
    tol = mp.mpf('1e-80')

    # Generate all positive roots using the standard algorithm:
    # Start with the simple roots; for each root alpha, compute
    # <alpha, alpha_i> (via Gram matrix). If positive, alpha + alpha_i
    # is also a root.
    # We represent roots as both coordinate vectors and as integer
    # coefficient vectors in the simple root basis.

    # Store positive roots as tuples of integer coefficients
    pos_root_coeffs = set()
    # Map from coefficient tuple to coordinate vector
    coord_map = {}

    # Initialize with simple roots
    queue = []
    for i in range(n):
        coeffs = [0] * n
        coeffs[i] = 1
        key = tuple(coeffs)
        pos_root_coeffs.add(key)
        coord_map[key] = list(simple_roots[i])
        queue.append(key)

    idx = 0
    while idx < len(queue):
        alpha_key = queue[idx]
        alpha_coords = coord_map[alpha_key]
        idx += 1

        for i in range(n):
            new_coeffs = list(alpha_key)
            new_coeffs[i] += 1
            new_key = tuple(new_coeffs)
            if new_key not in pos_root_coeffs:
                new_coords = add(alpha_coords, simple_roots[i])
                # A sum of positive roots with norm^2 = 2 is a positive root
                ns = norm_sq(new_coords)
                if abs(ns - root_norm_sq) < tol:
                    pos_root_coeffs.add(new_key)
                    coord_map[new_key] = new_coords
                    queue.append(new_key)

    # E6 has 36 positive roots
    assert len(pos_root_coeffs) == 36, f"Expected 36 positive roots, got {len(pos_root_coeffs)}"

    # All roots = positive roots ∪ negative roots
    all_roots = []
    for key in pos_root_coeffs:
        all_roots.append(coord_map[key])
        all_roots.append(neg(coord_map[key]))

    assert len(all_roots) == 72, f"Expected 72 roots, got {len(all_roots)}"

    # Normalize to unit vectors
    roots = []
    for v in all_roots:
        nv = mp.sqrt(dot(v, v))
        roots.append([v[k] / nv for k in range(n)])

    # Verify kissing constraint: all pairwise dot products <= 1/2
    m = len(roots)
    max_dot = mp.mpf('-1')
    min_dist = mp.mpf('inf')

    for i in range(m):
        # Check unit norm
        nv = mp.sqrt(dot(roots[i], roots[i]))
        if abs(nv - 1) > tol:
            raise ValueError(f"Non-unit vector at index {i}: norm = {nv}")

        for j in range(i + 1, m):
            d = dot(roots[i], roots[j])
            if d > max_dot:
                max_dot = d
            dist = mp.sqrt(mp.fsum((roots[i][k] - roots[j][k]) ** 2 for k in range(n)))
            if dist < min_dist:
                min_dist = dist

    if max_dot - mp.mpf('0.5') > tol:
        raise ValueError(f"Kissing constraint violated: max dot product = {max_dot}")
    if mp.mpf('1.0') - min_dist > tol:
        raise ValueError(f"Kissing constraint violated: min distance = {min_dist}")

    return mp.mpf(m)


if __name__ == "__main__":
    print(str(compute()))
