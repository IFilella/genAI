import torch
import psutil

def get_GPU_memory(device):
    current_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)
    max_reserved_memory = torch.cuda.max_memory_reserved(device)
    print('.........GPU Memory.........')
    print("Current GPU memory usage: %.2f MB" % (current_memory /
                                                 1024 / 1024))
    print("Reserved GPU memory: %.2f MB" % (reserved_memory /
                                            1024 / 1024))
    print("Max Reserved GPU memory: %.2f MB" %
          (max_reserved_memory / 1024 / 1024))
    free, total = torch.cuda.mem_get_info(device)
    print("Free %.2f MB " % (free / 1024 / 1024))
    print("Total %.2f MB" % (total / 1024 / 1024))
    print('...........................')

def get_CPU_memory():
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total
    available_memory = memory_info.available
    used_memory = memory_info.used
    memory_percent = memory_info.percent
    print('.........CPU Memory.........')
    print('Total CPU memory: %.2f MB' % (total_memory / 
                                         1024 / 1024))
    print('Current CPU memory usage: %.2f MB' % (used_memory /
                                                 1024 / 1024))
    print('Available CPU memory: %.2f MB' % (available_memory / 
                                             1024 / 1024))
def select_device(verbose=False):
    num_gpus = torch.cuda.device_count()
    max_perc = 0
    for i in range(num_gpus):
        free, total = torch.cuda.mem_get_info(device='cuda:%d' % i)
        perc = 100 - (((total - free) / total) * 100)
        if verbose:
            print('GPU %d: %.2f%%' % (i, perc))
        if perc >= max_perc:
            max_perc = perc
            gpu = i
    return('cuda:%d' % gpu)
