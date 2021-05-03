import os
import pickle
import shutil
import platform
import subprocess

__pickle_extension = '.pkl'
__open_cmd_dict = {'Linux': 'xdg-open',
                   'Darwin': 'open',
                   'Windows': 'start'}


def get_pythonpath():
    try:
        return os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        return []


def safe_remove(file):
    if os.path.exists(file):
        os.remove(file)
    else:
        pass


def start_open(file):
    open_cmd = __open_cmd_dict[platform.system()]
    subprocess.Popen([f'{open_cmd} {file}'], shell=True)


def save_object2txt(obj, file_name):
    if file_name[-4:] != '.txt' and '.' not in file_name:
        file_name += '.txt'

    with open(file_name, 'w') as f:
        f.write(''.join(["%s: %s\n" % (k, v) for k, v in obj.__dict__.items()]))


def save_pickle(obj, file):
    if file[-4:] != __pickle_extension:
        file += __pickle_extension

    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file):
    if file[-4:] != __pickle_extension:
        file += __pickle_extension

    with open(file, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def list_files(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


def safe_create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def split_directory_file(file):
    d, f = os.path.split(file)
    d += '/'
    return d, f


def ensure_final_slash(path):
    if path[-1] != '/':
        path += '/'
    return path


def ensure_initial_slash(path):
    if path[0] != '/':
        path = '/' + path
    return path


def ensure_initial_and_final_slash(path):
    path = ensure_initial_slash(path=path)
    path = ensure_final_slash(path=path)
    return path


def point_extension_wrapper(ext):
    if ext[0] != '.':
        ext = '.' + ext
    return ext


def ensure_file_extension(*, file, ext):
    ext = point_extension_wrapper(ext)

    if file[-len(ext)] != ext:
        idx_dot = file.find('.')
        if idx_dot != -1:
            file = file[:idx_dot]
        file += ext

    return file


def rel2abs_path(path, abs_directory):
    # abs_directory = '/Hello/HowAre/You/'
    # path = 'Hello/HowAre/You/good.txt'
    # path = 'good.txt'

    abs_directory = ensure_initial_slash(path=abs_directory)
    abs_directory = os.path.normpath(path=abs_directory)
    path = ensure_initial_slash(path=path)
    path = os.path.normpath(path=path)

    if abs_directory in path:
        return path
    else:
        return os.path.normpath(abs_directory + path)


def __read_head_tail(*, file, n=1, squeeze=True, head_or_tail):
    s = os.popen(f"{head_or_tail} -n {n} {file}").read()
    s = s.split('\n')[:-1]

    if squeeze and len(s) == 1:
        s = s[0]

    return s


def read_head(file, n=1, squeeze=True):
    return __read_head_tail(file=file, n=n, squeeze=squeeze, head_or_tail='head')


def read_tail(file, n=1, squeeze=True):
    return __read_head_tail(file=file, n=n, squeeze=squeeze, head_or_tail='tail')


def copy2clipboard(file):
    """
    https://apple.stackexchange.com/questions/15318/using-terminal-to-copy-a-file-to-clipboard
    -> works only for mac!
    """
    subprocess.run(['osascript',
                    '-e',
                    'set the clipboard to POSIX file "{}"'.format(file)])


# shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
def split_files_into_dirs(file_list, bool_fun, dir_list, base_dir=None, mode='dry'):

    if base_dir is not None:
        base_dir = os.path.normpath(base_dir)
    else:
        base_dir = ''

    if file_list is None and base_dir:
        file_list = os.listdir(base_dir)
        print(f'Get file_list from {base_dir}')

    for i, d_i in enumerate(dir_list):
        d_i = os.path.normpath(d_i)

        print(f"->{d_i}")

        j = 0
        while j < len(file_list):
            f_j = file_list[j]

            if bool_fun(f_j, i):
                f_j = os.path.normpath(f_j)
                f_j_new = f"{d_i}/{os.path.split(f_j)[-1]}"

                if mode == 'wet':
                    shutil.move(f"{base_dir}/{f_j}", f_j_new)
                print(f_j)

                file_list.pop(j)
            else:
                j += 1

    if mode != 'wet':
        print()
        print("'dry' mode is activated by default, to apply the changes use mode='wet')")
