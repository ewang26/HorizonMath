from mpmath import mp

mp.dps = 110

def compute():
    n = 5
    roots = []
    inv_sqrt2 = 1 / mp.sqrt(2)

    # D5 roots: all vectors with exactly two nonzero entries, each ±1, normalized to unit length
    for i in range(n):
        for j in range(i + 1, n):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = [mp.mpf('0') for _ in range(n)]
                    v[i] = mp.mpf(si) * inv_sqrt2
                    v[j] = mp.mpf(sj) * inv_sqrt2
                    roots.append(v)

    # Verify this is a valid kissing configuration for unit spheres around a central unit sphere:
    # centers are at radius 2, so after normalization to unit sphere we require pairwise distances >= 1
    # equivalently dot products <= 1/2.
    tol = mp.mpf('1e-80')

    def dot(a, b):
        return mp.fsum(a[k] * b[k] for k in range(n))

    def dist(a, b):
        return mp.sqrt(mp.fsum((a[k] - b[k]) ** 2 for k in range(n)))

    # Check norms
    for v in roots:
        nv = mp.sqrt(dot(v, v))
        if abs(nv - 1) > tol:
            raise ValueError("Non-unit vector encountered")

    max_dot = mp.mpf('-1')
    min_dist = mp.mpf('inf')

    m = len(roots)
    for i in range(m):
        for j in range(i + 1, m):
            d = dot(roots[i], roots[j])
            if d > max_dot:
                max_dot = d
            r = dist(roots[i], roots[j])
            if r < min_dist:
                min_dist = r

    if max_dot - mp.mpf('0.5') > tol:
        raise ValueError("Configuration violates kissing constraint (dot product too large)")
    if mp.mpf('1.0') - min_dist > tol:
        raise ValueError("Configuration violates kissing constraint (distance too small)")

    return mp.mpf(m)

if __name__ == "__main__":
    print(str(compute()))