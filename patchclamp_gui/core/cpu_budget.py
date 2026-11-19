import os


def available():
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if kernel.GetProcessAffinityMask(ctypes.c_void_p(-1), ctypes.byref(process_mask), ctypes.byref(system_mask)) and process_mask.value:
            return process_mask.value.bit_count()
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def budget(value, capacity):
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = 0
    return min(capacity, requested) if requested > 0 else max(1, capacity * 7 // 10)
