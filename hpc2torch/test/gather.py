import torch
import ctypes
import numpy as np
import argparse
import performance  
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

lib.my_gather.argtypes = [
    ctypes.c_void_p,  
    ctypes.c_void_p,  
    ctypes.c_void_p,  
    ctypes.c_long,   
    ctypes.c_long,    
    ctypes.c_long,    
    ctypes.c_long,   
    ctypes.c_long    
]

def test(inputShape, indexShape, axis, test_dtype, device):
    print(f"Testing Gather on {device} with x_shape:{inputShape} , indice_shape:{indexShape}, axis:{axis} ,dtype:{test_dtype}")
    
    inputTensor = torch.rand(inputShape, device=device, dtype=test_dtype)
    
    index_np = np.random.randint(0, inputShape[axis], indexShape).astype(np.int32)
    indexTensor = torch.from_numpy(index_np).to(torch.int64).to(device)
    
    def gather_fancy(inputTensor, axis, indexTensor):
        indices = [slice(None)] * inputTensor.dim()
        indices[axis] = indexTensor
        return inputTensor[tuple(indices)]
    
    outTensor = gather_fancy(inputTensor, axis, indexTensor)
    torch_gather_time = performance.CudaProfile((gather_fancy, (inputTensor, axis, indexTensor)))
    
    indexTensor_flat = indexTensor.contiguous().view(-1)
    index_num = indexTensor_flat.numel()
    
    rank = len(inputShape)
    outer = int(np.prod(inputShape[:axis])) if axis > 0 else 1
    mid   = inputShape[axis]
    inner = int(np.prod(inputShape[axis+1:])) if axis < rank - 1 else 1
    
    total_output = outer * index_num * inner
    Q_output = torch.zeros(total_output, device=device, dtype=test_dtype)
    
    input_ptr = ctypes.c_void_p(inputTensor.data_ptr())
    index_ptr = ctypes.c_void_p(indexTensor_flat.data_ptr())
    output_ptr = ctypes.c_void_p(Q_output.data_ptr())
    
    custom_gather_time = performance.CudaProfile((lib.my_gather, (input_ptr, index_ptr, output_ptr,
                                               outer, mid, inner, index_num, inputTensor.element_size())))
    
    performance.logBenchmark(torch_gather_time, custom_gather_time)
    
    Q_output = Q_output.view(outTensor.shape)
    
    tmpa = outTensor.cpu().numpy().flatten()
    tmpb = Q_output.cpu().numpy().flatten()
    atol = np.max(np.abs(tmpa - tmpb))
    rtol = atol / (np.max(np.abs(tmpb)) + 1e-8)
    
    print("absolute error:%.4e" % (atol))
    print("relative error:%.4e" % (rtol))

parser = argparse.ArgumentParser(description="Test gather on different devices.")
parser.add_argument('--device', default='cuda', help="Device to run the tests on.")
args = parser.parse_args()

if args.device == 'mlu':
    import torch_mlu

test_cases = [
    
    ((3, 2), (2, 2), 0, torch.float32, args.device),
    ((3, 2), (1, 2), 1, torch.float32, args.device),
    ((50257, 768), (16, 1024), 0, torch.float32, args.device),

    ((3, 2), (1, 2), 1, torch.float16, args.device),
    ((3, 2), (2, 2), 0, torch.float16, args.device),
    ((50257, 768), (16, 1024), 0, torch.float16, args.device),

    ((3, 2, 2), (3, 2, 2), 0, torch.float32, args.device),
    ((3, 2, 2), (3, 2, 2), 1, torch.float32, args.device),
    ((3, 2, 2), (3, 2, 2), 2, torch.float32, args.device),
    ((3, 2, 2, 2), (3, 2, 2, 5), 2, torch.float32, args.device)

]

filtered_test_cases = [
    (inp, ind, ax, dt, dev)
    for inp, ind, ax, dt, dev in test_cases if dev == args.device
]

for inputShape, indexShape, axis, test_dtype, device in filtered_test_cases:
    test(inputShape, indexShape, axis, test_dtype, device)
