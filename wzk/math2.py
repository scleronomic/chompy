import numpy as np
from scipy.stats import norm
from itertools import product

from wzk.numpy2 import shape_wrapper, axis_wrapper, insert
from wzk.dicts_lists_tuples import atleast_tuple

# a/b = (a+b) / a -> a / b =
golden_ratio = (np.sqrt(5.0) + 1) / 2


def number2digits(num):
    return [int(x) for x in str(num)]


def sin_cos(x):
    # https: // github.com / numpy / numpy / issues / 2626
    return np.sin(x), np.cos(x)


# Normalize
def normalize_01(x, low=None, high=None, axis=None):
    if low is None:
        low = np.min(x, axis=axis, keepdims=True)

    if high is None:
        high = np.max(x, axis=axis, keepdims=True)

    return (x-low) / (high-low)


def denormalize_01(x, low, high):
    return x * (high - low) + low


def normalize_11(x, low, high):
    """
    Normalize [low, high] to [-1, 1]
    low and high should either be scalars or have the same dimension as the last dimension of x
    """
    return 2 * (x - low) / (high - low) - 1


def denormalize_11(x, low, high):
    """
    Denormalize [-1, 1] to [low, high]
    low and high should either be scalars or have the same dimension as the last dimension of x
    """
    return (x + 1) * (high - low)/2 + low


def euclidean_norm(arr, axis=-1, squared=False):
    if squared:
        return (arr**2).sum(axis=axis)
    else:
        return np.sqrt((arr**2).sum(axis=axis))


def discretize(x, step):

    if np.isinf(step) or np.isnan(step):
        return x

    difference = x % step  # distance to the next discrete value

    if isinstance(x, (int, float)):
        if difference > step / 2:
            return x - (difference - step)
        else:
            return x - difference

    else:
        difference[difference > step / 2] -= step  # round correctly
        return x - difference


def d_linalg_norm__d_x(x, return_norm=False):
    """
    Last dimension is normalized.
    Calculate Jacobian
      xn       =           x * (x^2 + y^2 + z^2)^(-1/2)
    d xn / d x = (y^2 + z^2) * (x^2 + y^2 + z^2)^(-3/2)
    d yn / d y = (x^2 + y^2) * (x^2 + y^2 + z^2)^(-3/2)
    d zn / d z=  (x^2 + z^2) * (x^2 + y^2 + z^2)^(-3/2)

    Pattern of numerator
    X123
    0X23
    01X3
    012X

    d xn / d y = -(x*y) * (x^2 + y^2 + z^2)^(-3/2)
    d xn / d z = -(x*z) * (x^2 + y^2 + z^2)^(-3/2)

    jac = [[dxn/dx, dxn/dy, dxn/dz]
           [dyn/dx, dyn/dy, dyn/dz]
           [dzn/dx, dzn/dy, dzn/dz]

    """

    n = x.shape[-1]

    off_diag_idx = [[j for j in range(n) if i != j] for i in range(n)]

    jac = np.empty(x.shape + x.shape[-1:])
    x_squared = x**2

    # Diagonal
    jac[:, np.arange(n), np.arange(n)] = x_squared[..., off_diag_idx].sum(axis=-1)

    # Off-Diagonal
    jac[:, np.arange(n)[:, np.newaxis], off_diag_idx] = -x[..., np.newaxis] * x[:, off_diag_idx]

    jac *= (x_squared.sum(axis=-1, keepdims=True)**(-3/2))[..., np.newaxis]

    if return_norm:
        x /= np.sqrt(x_squared.sum(axis=-1, keepdims=True))
        return x, jac
    else:
        return jac


# Smooth
def smooth_step(x):
    """https://en.wikipedia.org/wiki/Smoothstep
    Interpolation which has zero 1st-order derivatives at x = 0 and x = 1,
     ~ cubic Hermite interpolation with clamping.
    """
    res = -2 * x**3 + 3 * x**2
    return np.clip(res, 0, 1)


def smoother_step(x):
    """https://en.wikipedia.org/wiki/Smoothstep+
    Ken Perlin suggests an improved version of the smooth step function,
    which has zero 1st- and 2nd-order derivatives at x = 0 and x = 1"""
    res = +6 * x**5 - 15 * x**4 + 10 * x**3
    return np.clip(res, 0, 1)


# Divisors
def divisors(n, with_1_and_n=False):
    """
    https://stackoverflow.com/questions/171765/what-is-the-best-way-to-get-all-the-divisors-of-a-number#171784
    """

    # Get factors and their counts
    factors = {}
    nn = n
    i = 2
    while i*i <= nn:
        while nn % i == 0:
            if i not in factors:
                factors[i] = 0
            factors[i] += 1
            nn //= i
        i += 1
    if nn > 1:
        factors[nn] = 1

    primes = list(factors.keys())

    # Generates factors from primes[k:] subset
    def generate(k):
        if k == len(primes):
            yield 1
        else:
            rest = generate(k+1)
            prime = primes[k]
            for _factor in rest:
                prime_to_i = 1
                # Prime_to_i iterates prime**o values, o being all possible exponents
                for _ in range(factors[prime] + 1):
                    yield _factor * prime_to_i
                    prime_to_i *= prime

    if with_1_and_n:
        return list(generate(0))
    else:
        return list(generate(0))[1:-1]


def get_mean_divisor_pair(n):
    """
    Calculate the 'mean' pair of divisors. The two divisors should be as close as possible to the sqrt(n).
    The smaller divisor is the first value of the output pair
    10 -> 2, 5
    20 -> 4, 5
    24 -> 4, 6
    25 -> 5, 5
    30 -> 5, 6
    40 -> 5, 8
    """
    assert isinstance(n, int)
    assert n >= 1

    div = divisors(n)
    if n >= 3 and len(div) == 0:  # Prime number -> make at least even
        return 1, n

    div.sort()

    # if numbers of divisors is odd -> n = o * o : power number
    if len(div) % 2 == 1:
        idx_center = len(div) // 2
        return div[idx_center], div[idx_center]

    # else get the two numbers at the center
    else:
        idx_center_plus1 = len(div) // 2
        idx_center_minus1 = idx_center_plus1 - 1
        return div[idx_center_minus1], div[idx_center_plus1]


def get_divisor_safe(numerator, denominator):
    divisor = numerator / denominator
    divisor_int = int(divisor)

    assert divisor_int == divisor
    return divisor_int


def doubling_factor(small, big):
    return np.log2(big / small)


def modulo(x, low, high):
    return (x - low) % (high - low) + low


def angle2minuspi_pluspi(x):
    return modulo(x=x, low=-np.pi, high=+np.pi)
    # modulo is faster for larger arrays, for small ones they are similar but arctan is faster in this region
    #  -> as always you have to make an trade-off
    # return np.arctan2(np.sin(x), np.cos(x))


# Derivative
def numeric_derivative(*, fun, x, eps=1e-5, axis=-1,
                       **kwargs_fun):
    """
    Use central difference scheme to calculate the
    numeric derivative of fun at point x.
    Axis indicates the dimensions of the free variables.
    The result has the shape f(x).shape + (x.shape)[axis]
    """
    axis = axis_wrapper(axis=axis, n_dim=x.ndim)

    fun_shape = np.shape(fun(x, **kwargs_fun))
    var_shape = atleast_tuple(np.array(np.shape(x))[axis])
    derv = np.empty(fun_shape + var_shape)

    eps_mat = np.empty_like(x, dtype=float)

    def update_eps_mat(_idx):
        eps_mat[:] = 0
        insert(eps_mat, val=eps, idx=_idx, axis=axis)

    for idx in product(*(range(s) for s in var_shape)):
        update_eps_mat(_idx=idx)
        derv[(Ellipsis,) + idx] = (fun(x + eps_mat, **kwargs_fun) - fun(x - eps_mat, **kwargs_fun)) / (2 * eps)

    return derv


# Statistics for distribution of number of obstacles
def p_normal_skew(x, loc=0.0, scale=1.0, a=0.0):
    t = (x - loc) / scale
    return 2 * norm.pdf(t) * norm.cdf(a*t)


def normal_skew_int(loc=0.0, scale=1.0, a=0.0, low=None, high=None, size=1):
    if low is None:
        low = loc-10*scale
    if high is None:
        high = loc+10*scale+1

    p_max = p_normal_skew(x=loc, loc=loc, scale=scale, a=a)

    samples = np.zeros(np.prod(size))

    for i in range(int(np.prod(size))):
        while True:
            x = np.random.randint(low=low, high=high)
            if np.random.rand() <= p_normal_skew(x, loc=loc, scale=scale, a=a) / p_max:
                samples[i] = x
                break

    samples = samples.astype(int)
    if size == 1:
        samples = samples[0]
    return samples


def random_uniform_ndim(*, low, high, shape=None):
    n_dim = np.shape(low)[0]
    x = np.zeros(shape_wrapper(shape) + (n_dim,))
    for i in range(n_dim):
        x[..., i] = np.random.uniform(low=low[i], high=high[i], size=shape)
    return x


def get_stats(x, axis=None, return_array=False):
    stats = {'mean': np.mean(x, axis=axis),
             'std':  np.std(x, axis=axis),
             'median': np.median(x, axis=axis),
             'min': np.min(x, axis=axis),
             'max': np.max(x, axis=axis)}

    if return_array:
        return np.array([stats['mean'], stats['std'], stats['median'], stats['min'], stats['max']])

    return stats


# Magic
def magic(n):
    """
    Equivalent of the MATLAB function:
    M = magic(n) returns an n-by-n matrix constructed from the integers 1 through n2 with equal row and column sums.
    https://stackoverflow.com/questions/47834140/numpy-equivalent-of-matlabs-magic
    """

    n = int(n)

    if n < 1:
        raise ValueError('Size must be at least 1')
    if n == 1:
        return np.array([[1]])
    elif n == 2:
        return np.array([[1, 3], [4, 2]])
    elif n % 2 == 1:
        p = np.arange(1, n+1)
        return n*np.mod(p[:, None] + p - (n+3)//2, n) + np.mod(p[:, None] + 2*p-2, n) + 1
    elif n % 4 == 0:
        j = np.mod(np.arange(1, n+1), 4) // 2
        k = j[:, None] == j
        m = np.arange(1, n*n+1, n)[:, None] + np.arange(n)
        m[k] = n*n + 1 - m[k]
    else:
        p = n//2
        m = magic(p)
        m = np.block([[m, m+2*p*p], [m+3*p*p, m+p*p]])
        i = np.arange(p)
        k = (n-2)//4
        j = np.concatenate((np.arange(k), np.arange(n-k+1, n)))
        m[np.ix_(np.concatenate((i, i+p)), j)] = m[np.ix_(np.concatenate((i+p, i)), j)]
        m[np.ix_([k, k+p], [0, k])] = m[np.ix_([k+p, k], [0, k])]
    return m


# Geometry

def get_dcm2d(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    dcm = np.array([[c, -s],
                    [s, c]])

    # Make sure the 2x2 matrix is at the last 2 dimensions of the array, even if theta was multidimensional
    np.moveaxis(np.moveaxis(dcm, 0, -1), 0, -1)
    return dcm


def distance_line_point(x0, x1, x2):
    """
    http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    Distance between x0 and the line defined by {x1 + a*x2}
    """
    return np.linalg.norm(np.cross(x0-x1, x0-x2), axis=-1) / np.linalg.norm(x2-x1, axis=-1)


def circle_circle_intersection(xy0, r0, xy1, r1):
    """
    https://stackoverflow.com/a/55817881/7570817
    https://mathworld.wolfram.com/Circle-CircleIntersection.html

    circle 1: (x0, y0), radius r0
    circle 2: (x1, y1), radius r1
    """

    d = np.linalg.norm(xy1 - xy0)

    # non intersecting
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0 - r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(r0 ** 2 - a ** 2)
        d01 = (xy1 - xy0) / d

        xy2 = xy0 + a * d01[::+1] * [+1, +1]
        xy3 = xy2 + h * d01[::-1] * [+1, -1]
        xy4 = xy2 + h * d01[::-1] * [-1, +1]

        return xy3, xy4


def ray_sphere_intersection(rays, spheres):
    """
    :param rays: n_rays x 2 x 3    (axis=1: origin, target)
    :param spheres: n_spheres x 4  (axis=1: x, y, z, r)
    :return: n_rays x n_spheres (boolean array) with res[o, j] = True if ray o intersects with sphere j
    Formula from: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    """

    o = rays[:, 0]
    u = np.diff(rays, axis=1)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)

    c = spheres[:, :3]
    r = spheres[:, 3:].T
    co = (o[:, np.newaxis, :] - c[np.newaxis, :, :])
    res = (u * co).sum(axis=-1)**2 - (co**2).sum(axis=-1) + r**2
    return res >= 0


def ray_sphere_intersection_2(rays, spheres, r):
    """
    :param rays: n x n_rays x 2 x 3    (axis=2: origin, target)
    :param spheres: n x n_spheres x 3  (axis=2: x, y, z)
    :param r: n_spheres
    :return: n x n_rays x n_spheres (boolean array) with res[:, o, j] = True if ray o intersects with sphere j
    Formula from: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection

    rays = np.random.random((10, 4, 2, 3))
    spheres = np.random.random((10, 5, 3))
    r = np.ones(5) * 0.1
    res = ray_sphere_intersection_2(rays=rays, spheres=spheres, r=r)
    """

    o = rays[:, :, 0]
    u = np.diff(rays, axis=-2)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)

    c = spheres[..., :3]
    co = (o[:, :, np.newaxis, :] - c[:, np.newaxis, :, :])
    res = (u * co).sum(axis=-1)**2 - (co**2).sum(axis=-1) + r**2
    return res >= 0


def sample_points_on_disc(radius, size=None):
    rho = np.sqrt(np.random.uniform(0, radius**2, size=size))
    theta = np.random.uniform(0, 2*np.pi, size)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    return x, y


def sample_points_on_sphere_3d(size):
    size = shape_wrapper(shape=size)
    x = np.empty(tuple(size) + (3,))
    theta = np.random.uniform(low=0, high=2*np.pi, size=size)
    phi = np.arccos(1-2*np.random.uniform(low=0, high=1, size=size))
    sin_phi = np.sin(phi)
    x[..., 0] = sin_phi * np.cos(theta)
    x[..., 1] = sin_phi * np.sin(theta)
    x[..., 2] = np.cos(phi)

    return x


def sample_points_on_sphere_nd(size, n_dim, ):

    # if np.shape(shape) < 2:
    #     safety = 100
    # else:

    safety = 1.2

    size = shape_wrapper(shape=size)
    volume_sphere = hyper_sphere_volume(n_dim)
    volume_cube = 2**n_dim
    safety_factor = int(np.ceil(safety * volume_cube/volume_sphere))

    size_w_ndim = size + (n_dim,)
    size_sample = (safety_factor,) + size_w_ndim

    x = np.random.uniform(low=-1, high=1, size=size_sample)
    x_norm = np.linalg.norm(x, axis=-1)
    bool_keep = x_norm < 1
    n_keep = bool_keep.sum()
    # print(n_keep / np.shape(shape))
    assert n_keep > np.size(size)
    raise NotImplementedError


def hyper_sphere_volume(n_dim, r=1.):
    """https: // en.wikipedia.org / wiki / Volume_of_an_n - ball"""
    n2 = n_dim//2
    if n_dim % 2 == 0:
        return (np.pi ** n2) / np.math.factorial(n2) * r**n_dim
    else:
        return 2*(np.math.factorial(n2)*(4*np.pi)**n2) / np.math.factorial(n_dim) * r**n_dim


# Clustering
def k_farthest_neighbors(x, k, weighting=None):
    n = len(x)

    m_dist = x[np.newaxis, :, :] - x[:, np.newaxis, :]
    weighting = np.ones(x.shape[-1]) if weighting is None else weighting
    # m_dist = np.linalg.norm(m_dist * weighting, axis=-1)
    m_dist = ((m_dist * weighting)**2).sum(axis=-1)

    cum_dist = m_dist.sum(axis=-1)

    idx = [np.argmax(cum_dist)]

    for i in range(k-1):
        m_dist_cur = m_dist[idx]
        m_dist_cur_sum = m_dist_cur.sum(axis=0)
        # m_dist_cur_std = np.std(m_dist_cur, axis=0)
        obj = m_dist_cur_sum   # + (m_dist_cur_std.max() - m_dist_cur_std) * 1000
        idx_new = np.argsort(obj)[::-1]
        for j in range(n):
            if idx_new[j] not in idx:
                idx.append(idx_new[j])
                break

    return np.array(idx)


def test_k_farthest_neighbors():
    x = np.random.random((200, 2))
    k = 10
    idx = k_farthest_neighbors(x=x, k=k)

    from wzk import new_fig
    fig, ax = new_fig(aspect=1)
    ax.plot(*x.T, ls='', marker='o', color='b', markersize=5, alpha=0.5)
    ax.plot(*x[idx, :].T, ls='', marker='x', color='r', markersize=10)


# Combinatorics
def binomial(n, k):
    return np.math.factorial(n) // np.math.factorial(k) // np.math.factorial(n - k)


def random_subset(n, k, m, dtype=np.uint16):
    assert n == np.array(n, dtype=dtype)
    return np.array([np.random.choice(n, k, replace=False) for _ in range(m)]).astype(np.uint16)

