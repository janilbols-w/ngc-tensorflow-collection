#!/usr/bin/env python

import os

LOCAL_DEBUG = os.environ.get('LOCAL_DEBUG', None) is not None
IS_CI_JOB = os.environ.get("CI_JOB_ID", None) is not None

DATETIME_STR_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

if IS_CI_JOB or LOCAL_DEBUG:
    VERBOSE_MODE = True
else:
    VERBOSE_MODE = False


def _print_err(err):
    _FAIL_CLR = '\033[91m'
    _END_CLR = '\033[0m'
    
    if VERBOSE_MODE:
        print(_FAIL_CLR + err + _END_CLR)


def _exec_cmd(cmd):
    import subprocess

    try:
        ps = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        out = ps.communicate()[0].decode().strip()
        if ps.returncode == 0:
            return out
        else:
            err_txt = (
                "Error raised during process "
                "[CODE: {}]: {}".format(ps.returncode, out))
            _print_err("RuntimeError: " + err_txt)
            raise RuntimeError(err_txt)
    except subprocess.CalledProcessError as e:
        _print_err("subprocess.CalledProcessError: " + str(e))
        raise RuntimeError() from e


def _localize_datetime(timestamp):
    import pytz
    timezone = pytz.timezone("America/Los_Angeles")
    return timezone.localize(timestamp)


def _get_datetime_now(as_string=False):
    import datetime
    timestamp = datetime.datetime.now()
    timestamp = _localize_datetime(timestamp)

    if as_string:
        return _get_str_from_datetime(timestamp)
    else:
        return timestamp


def _get_datetime_from_str(timestamp_str):
    import datetime
    return _localize_datetime(datetime.datetime.strptime(
        timestamp_str, DATETIME_STR_FORMAT
    ))


def _get_str_from_datetime(timestamp):
    return timestamp.strftime(DATETIME_STR_FORMAT)


def get_cpu_info():
    try:
        import psutil
        cpu_thread_count = psutil.cpu_count(logical=True)
        core_count_per_socket = psutil.cpu_count(logical=False)
    except ModuleNotFoundError:
        _print_err("`psutil` is not installed. Please run `pip install psutil`")
        import multiprocessing
        cpu_thread_count = multiprocessing.cpu_count()
        core_count_per_socket = 0

    try:
        cpu_socket_count = _exec_cmd(
            'cat /proc/cpuinfo | grep \"physical id\" | sort -u | wc -l'
        )
    except RuntimeError:
        cpu_socket_count = 1

    core_count_total = int(core_count_per_socket) * int(cpu_socket_count)

    import platform
    cpu_arch = platform.uname().processor

    try:
        cpu_model = _exec_cmd(
            'cat /proc/cpuinfo | grep "model name" -m1'
        ).split(": ")[-1]
    except RuntimeError:
        cpu_model = "Unknown CPU"
    return {
        "core_count_per_socket": core_count_per_socket,
        "core_count_total": core_count_total,
        "thread_count": cpu_thread_count,
        "cpu_socket_count": cpu_socket_count,
        "arch": cpu_arch,
        "name": cpu_model
    }


def get_ram_capacity(round_decimal=2):
    import os
    import math

    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    mem_gib = mem_bytes / (1024.**3)
    round_factor = 10 ** round_decimal
    mem_gib = math.ceil(mem_gib * round_factor) / round_factor
    return mem_gib


def get_git_information():
    import os

    def get_commit_hash():
        commit = os.environ.get("NVIDIA_BUILD_REF", None)

        if commit is None:
            try:
                commit = _exec_cmd('git rev-parse HEAD')
            except RuntimeError:
                commit = "0000000000000000000000000000000000000000"

        return commit

    def get_git_branch():
        branch = os.environ.get("COMMIT_BRANCH", None)

        if branch is None:
            try:
                branch = _exec_cmd('git rev-parse --abbrev-ref HEAD')
            except RuntimeError:
                branch = "unknown"

        return branch

    def get_git_submodule_commit_hash():
        commit = os.environ.get("COMMIT_TF_SUBMODULE_SHA", None)

        if commit is None:
            try:
                commit = _exec_cmd('git rev-parse @:./tensorflow-source')
            except RuntimeError:
                commit = "0000000000000000000000000000000000000000"
                
        return commit

    return {
        "dl/dgx/tf": {"commit": get_commit_hash(), "branch": get_git_branch()},
        "dl/tf/tf": {"commit": get_git_submodule_commit_hash()}
    }


def get_tf_version():
    try:
        version = _exec_cmd(
            "grep '_VERSION =' "
            "tensorflow-source/tensorflow/tools/pip_package/setup.py"
        )[12:-1]
    except RuntimeError:
        version = "0.0.0"
    return version


def get_nvidia_version_info():
    import os
    try:
        cuda_version = os.environ.get("CUDA_VERSION", None)
        if cuda_version is None:
            cuda_version = _exec_cmd(
                "nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-"
            )
    except RuntimeError:
        cuda_version = "00.0.0"

    try:
        driver_version = os.environ.get("CUDA_DRIVER_VERSION", None)
        if driver_version is None:
            driver_version = _exec_cmd(
                "nvidia-smi | grep 'Driver Version' | awk '{print $6}' "
                "| cut -c1-"
            )
    except RuntimeError:
        driver_version = "000.00.00"

    return {
        "cuda_ver": cuda_version,
        "driver_version": driver_version,
    }


def get_distro_info():
    try:
        distro = _exec_cmd("cat /etc/os-release | grep VERSION=").split("\"")[1]
    except RuntimeError:
        distro = "Unknown"
    return distro


def get_python_version():
    import sys
    ver = sys.version_info
    return "{major}.{minor}.{micro}".format(
        major=ver.major,
        minor=ver.minor,
        micro=ver.micro,
    )


def get_bazel_version():
    try:
        commit = _exec_cmd("bazel --version").split(" ")[-1]
    except RuntimeError:
        commit = "0.0.0"
    return commit


def get_build_time():
    import os
    start_time = os.environ.get("BUILD_START_TIMESTAMP", '')

    if start_time == "":
        _print_err("Env Var: `BUILD_START_TIMESTAMP` was not defined")
        return -1
    else:
        start_time = _get_datetime_from_str(start_time)

    try:
        return int((_get_datetime_now() - start_time).total_seconds())
    except ValueError as e:
        _print_err(
            "Value Error: `BUILD_START_TIMESTAMP` is most likely invalid: "
            "`{}`. Err: {}".format(start_time, str(e))
        )
        return -1


def collect_data():
    import os
    cpu_info = get_cpu_info()
    git_data = get_git_information()
    nvinfo = get_nvidia_version_info()

    data = {
        "bazel_cache_used": (
            os.environ.get("BAZEL_CACHE_FLAG", None) is not None
        ),
        "bazel_version": get_bazel_version(),
        "git_dl/dgx/tf_branch": git_data["dl/dgx/tf"]["branch"],
        "git_dl/dgx/tf_commit": git_data["dl/dgx/tf"]["commit"],
        "git_dl/tf/tf_commit": git_data["dl/tf/tf"]["commit"],
        "machine_cpu_arch": cpu_info["arch"],
        "machine_cpu_core_count_per_socket": cpu_info["core_count_per_socket"],
        "machine_cpu_core_count_total": cpu_info["core_count_total"],
        "machine_cpu_name": cpu_info["name"],
        "machine_cpu_socket_count": cpu_info["cpu_socket_count"],
        "machine_cpu_thread_count": cpu_info["thread_count"],
        "machine_ram_capacity_in_gb": get_ram_capacity(),
        "nvidia_build_id": os.environ.get("CI_JOB_ID", ""),
        "nvidia_cuda_version": nvinfo["cuda_ver"],
        "nvidia_driver_version": nvinfo["driver_version"],
        "os_version": get_distro_info(),
        "python_version": get_python_version(),
        "tf_version": get_tf_version(),
        "timestamp": _get_datetime_now(as_string=True),
        "total_build_time_in_sec": get_build_time(),
    }

    return {key: str(val) for key, val in data.items()}


if __name__ == "__main__":

    import os
    import sys
    import requests
    from pprint import pprint

    data = collect_data()

    if VERBOSE_MODE:
        pprint(data)

    if LOCAL_DEBUG:
        ip_addr = "localhost"
    else:
        ip_addr = '10.31.241.12'

    endpoint_url = 'http://{}:9999/api/submit_job_results'.format(ip_addr)

    if VERBOSE_MODE:
        print("DEBUG: Endpoint URL:", endpoint_url)

    try:
        result = requests.post(endpoint_url, json=data)
        returned_data = result.json()

        if result.ok:
            assert (returned_data == data)
            if VERBOSE_MODE:
                print("[*] Data successfully uploaded!")
            sys.exit(0)
        else:
            print('Error [CODE: {}] - {}'.format(
                result.status_code, returned_data
            ))
            sys.exit(1)

    except (
        requests.exceptions.ConnectTimeout,
        requests.exceptions.ConnectionError
    ) as e:
        if VERBOSE_MODE:
            print("Server is not reachable: {}".format(str(e)))
        sys.exit(1)
