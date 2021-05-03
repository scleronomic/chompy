import json
import argparse
import collections
import numpy as np


class ObjectDict:
    def to_dict(self):
        return self.__dict__

    def update(self, new_dict):
        if isinstance(new_dict, ObjectDict):
            self.__dict__.update(new_dict.__dict__)
        else:
            self.__dict__.update(new_dict)

    def copy(self):
        new = ObjectDict()
        new.update(self)
        return new


def totuple(a):
    if isinstance(a, str) and len(a) == 1:
        return a
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def tolist(a):
    if isinstance(a, str) and len(a) == 1:
        return a
    try:
        return list(tolist(i) for i in a)
    except TypeError:
        return a


def tuple_extract(t, default, mode='use_default'):
    """
    default: tuple

    mode: 'use_default'
          'repeat'
    """

    if t is None:
        return default

    t = atleast_tuple(t)
    length_default = len(default)
    length = len(t)

    if length == length_default:
        return t

    if mode == 'use_default':
        return t + default[length:]
    elif mode == 'repeat':
        assert length == 1, "mode 'repeat' expects length(t) == 1"
        return t * length_default

    else:
        raise ValueError(f"Unknown mode {mode}")


def remove_nones(lst):
    return [item for item in lst if item is not None]


def atleast_list(*lists, convert=True):
    """
    adapted from numpy.atleast_1d
    """
    res = []
    for lst in lists:
        if not isinstance(lst, list):
            if convert:
                lst = tolist(lst)
                if not isinstance(lst, list):
                    lst = [lst]
            else:
                lst = [lst]

        res.append(lst)

    if len(res) == 1:
        return res[0]
    else:
        return res


def atleast_tuple(*tuples, convert=True):
    """
    adapted from numpy.atleast_1d
    """
    res = []
    for tpl in tuples:
        if not isinstance(tpl, tuple):
            if convert:
                tpl = totuple(tpl)
                if not isinstance(tpl, tuple):
                    tpl = (tpl,)

            else:
                tpl = (tpl,)
        res.append(tpl)

    if len(res) == 1:
        return res[0]
    else:
        return res


# Slice and range
def range2slice(r):
    return slice(r.start, r.stop, r.step)


def slice2range(s):
    return range(0 if s.start is None else s.start,
                 s.stop,
                 1 if s.step is None else s.step)


def __slice_or_range2tuple(sor, type2):
    if type2 == 'slice':
        default1, default2, default3 = None, None, None
    elif type2 == 'range':
        default1, default2, default3 = 0, 1, 1
    else:
        raise ValueError(f"Unknown {type2}")

    if isinstance(sor, (slice, range)):
        return sor.start, sor.stop, sor.step

    elif isinstance(sor, int):
        return default1, sor, default3

    elif sor is None:
        return default1, default2, default3

    elif isinstance(sor, tuple):
        if len(sor) == 1:
            return default1, sor[0], default3
        elif len(sor) == 2:
            return sor[0], sor[1], default3
        elif len(sor) == 3:
            return sor
        else:
            raise ValueError('tuple must be have length={1, 2, 3}')
    else:
        raise TypeError('r must be {slice, range, int, tuple}')


def slice2tuple(s):
    return __slice_or_range2tuple(s, 'slice')


def range2tuple(r):
    return __slice_or_range2tuple(r, 'range')


def slice_add(a, b):
    a = slice2tuple(a)
    b = slice2tuple(b)
    return slice(a[0]+b[0], a[1]+b[1], max(a[2], b[2]))


def range_add(a, b):
    a = range2tuple(a)
    b = range2tuple(b)
    return range(a[0]+b[0], a[1]+b[1], max(a[2], b[2]))


def tl_add(a, b):
    """
    Element-wise addition for tuples or lists.
    """

    lst = [aa + bb for aa, bb in zip(a, b)]

    if type(a) == list:
        return lst
    elif type(a) == tuple:
        return tuple(lst)
    else:
        raise ValueError(f"Unknown type {type(a)}")


def depth_list(lst):
    return isinstance(lst, list) and max(map(depth_list, lst)) + 1


def depth_tuple(tpl):
    return isinstance(tpl, tuple) and max(map(depth_tuple, tpl)) + 1


def flatten_gen(lst, __cur_depth=0, max_depth=100):

    for el in lst:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            __cur_depth += 1
            if __cur_depth <= max_depth:
                yield from flatten_gen(el, __cur_depth=__cur_depth, max_depth=max_depth)
            else:
                yield el
            __cur_depth -= 1

        else:
            yield el


def flatten(lst, max_depth=100):
    return list(flatten_gen(lst=lst, max_depth=max_depth))


def element_at_depth_gen(lst, depth=0, with_index=False, __cur_depth=0):

    def __yield1(ii, ele):
        if with_index:
            return (ii,), ele
        else:
            return el

    def __yield2(ii, ele):
        if with_index:
            return ((ii,) + ele[0]), ele[1]
        else:
            return el

    for i, el in enumerate(lst):

        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            __cur_depth += 1
            if __cur_depth < depth:
                # better yield from ...
                for el2 in element_at_depth_gen(el, depth=depth, with_index=with_index, __cur_depth=__cur_depth):
                    yield __yield2(i, el2)

            else:  # __cur_depth == depth
                yield __yield1(i, el)
            __cur_depth -= 1

        else:
            if __cur_depth == depth:
                yield __yield1(i, el)


def element_at_depth(lst, depth=0, with_index=False):
    return list(element_at_depth_gen(lst=lst, depth=depth, with_index=with_index))


def change_tuple_order(tpl):
    return tuple(map(lambda *tt: tuple(tt), *tpl))


def change_list_order(lst):
    return list(map(lambda *ll: list(ll), *lst))


def get_first_non_empty(lst):
    """
    lst = [[], [], 1, [2, 3, 4], [], []] -> 1
    lst = [[], [], False, (), [2, 3, 4], [], []] -> [2, 3, 4]
    """
    for element in lst:
        if element:
            return element


def repeat_dict(d, n):
    d_repeat = {}

    for i in range(n):
        d_i = {}
        for key in d:
            if isinstance(d[key], (tuple, list, np.ndarray)):
                d_i[key] = d[key][i]
            else:
                d_i[key] = d[key]

        d_repeat[i] = d_i
    return d_repeat


def list_allclose(a, b):
    if isinstance(a, (tuple, list)):
        return np.array([np.allclose(aa, bb) for aa, bb in zip(a, b)])
    else:
        return np.allclose(a, b)


# json
def write_dict2json(file, d, **kwargs):
    with open(file=file, mode='w') as f:
        f.write(json.dumps(d, **kwargs))


def read_json2dict(file):
    with open(file=file, mode='r') as f:
        d = json.load(f)
    return d


# Dicts
def rename_dict_keys(d, new_keys_dict):
    for old_k in new_keys_dict:
        d[new_keys_dict[old_k]] = d.pop(old_k)



