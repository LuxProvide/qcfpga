import os
import sys
import random
import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array
from pathlib import Path
from pyopencl.reduction import ReductionKernel
from collections import defaultdict


# Setup the OpenCL Context here to not prompt every execution
platform = None
context  = None
device   = None
program  = None


class Backend:
    """
    A class for the OpenCL backend to the simulator.

    This class shouldn't be used directly, as many of the
    methods don't have the same input checking as the State
    class.
    """

    # @profile
    def __init__(self, num_qubits, dtype=np.complex64):
        if not context:
            create_context()
        
        """
        Initialize a new OpenCL Backend

        Takes an argument of the number of qubits to use
        in the register, and returns the backend.
        """
        self.num_qubits = num_qubits
        self.dtype = dtype

        self.queue = cl.CommandQueue(context)


        # Buffer for the state vector
        self.buffer = pycl_array.to_device(
            self.queue,
            np.eye(1, 2**num_qubits, dtype=dtype)
        )



    def apply_gate(self, gate, target):
        """Applies a gate to the quantum register"""
        program.apply_gate(
            self.queue,
            [int(2**self.num_qubits / 2)],
            None,
            self.buffer.data,
            np.int32(target),
            self.dtype(gate.a),
            self.dtype(gate.b),
            self.dtype(gate.c),
            self.dtype(gate.d)
        )

    def apply_controlled_gate(self, gate, control, target):
        """Applies a controlled gate to the quantum register"""

        program.apply_controlled_gate(
            self.queue,
            [int(2**self.num_qubits / 2)],
            None,
            self.buffer.data,
            np.int32(control),
            np.int32(target),
            self.dtype(gate.a),
            self.dtype(gate.b),
            self.dtype(gate.c),
            self.dtype(gate.d)
        )
    
    def apply_controlled_controlled_gate(self, gate, control1, control2, target):
        """Applies a controlled controlled gate (such as a toffoli gate) to the quantum register"""

        program.apply_controlled_controlled_gate(
            self.queue,
            [int(2**self.num_qubits / 2)],
            None,
            self.buffer.data,
            np.int32(control1),
            np.int32(control2),
            np.int32(target),
            self.dtype(gate.a),
            self.dtype(gate.b),
            self.dtype(gate.c),
            self.dtype(gate.d)
        )

    def seed(self, val):
        random.seed(val)
        
    def measure(self, samples=1):
        """Measure the state of a register"""
        # This is a really horrible method that needs a rewrite - the memory
        # is attrocious

        probabilities = self.probabilities()
        # print(probabilities)
        # print(np.sum(self.amplitudes()))
        choices = np.random.choice(
            np.arange(0, 2**self.num_qubits), 
            samples, 
            p=probabilities
        )
        
        results = defaultdict(int)
        for i in choices:
            results[np.binary_repr(i, width=self.num_qubits)] += 1
        
        return dict(results)

    def measure_first(self, num, samples):
        probabilities = self.probabilities()
        # print(probabilities)
        # print(np.sum(self.amplitudes()))
        choices = np.random.choice(
            np.arange(0, 2**self.num_qubits), 
            samples, 
            p=probabilities
        )
        
        results = defaultdict(int)
        for i in choices:
            key = np.binary_repr(i, width=self.num_qubits)[-num:]
            results[key] += 1
        
        return dict(results)
       

    def qubit_probability(self, target):
        """Get the probability of a single qubit begin measured as '0'"""

        out = pycl_array.to_device(
            self.queue,
            np.zeros(2**self.num_qubits, dtype=np.float32)
        )

        program.probability_single(
            self.queue,
           [int(2**self.num_qubits)],
            None,
            self.buffer.data,
            out.data,
            np.int32(target)

        )

        return np.sum(out.get())
        
    def reset(self, target):
        probability_of_0 = self.qubit_probability(target)
        norm = 1 / np.sqrt(probability_of_0)
        
        program.collapse(
            self.queue,
            [int(2**self.num_qubits)],
            # 2**self.num_qubits,
            None,
            self.buffer.data,
            np.int32(target),
            np.int32(0),
            np.float32(norm)
        )

    def measure_collapse(self, target):
        probability_of_0 = self.qubit_probability(target)
        random_number = random.random()

        if random_number <= probability_of_0:
            outcome = '0'
            norm = 1 / np.sqrt(probability_of_0)
        else:
            outcome = '1'
            norm = 1 / np.sqrt(1 - probability_of_0)

        program.collapse(
            self.queue,
            [int(2**self.num_qubits)],
            # 2**self.num_qubits,
            None,
            self.buffer.data,
            np.int32(target),
            np.int32(outcome),
            np.float32(norm)
        )
        return outcome

    def measure_qubit(self, target, samples):
        probability_of_0 = self.qubit_probability(target)

        choices = np.random.choice(
            [0, 1], 
            samples, 
            p=[probability_of_0, 1-probability_of_0]
        )
        
        results = defaultdict(int)
        for i in choices:
            results[np.binary_repr(i, width=1)] += 1
        
        return dict(results)

    def single_amplitude(self, i):
        """Gets a single probability amplitude"""
        out = pycl_array.to_device(
            self.queue,
            np.empty(1, dtype=np.complex64)
        )

        program.get_single_amplitude(
            self.queue, 
            (1,),
            None, 
            self.buffer.data,
            out.data,
            np.int32(i)
        )

        return out[0]

    def amplitudes(self):
        """Gets the probability amplitudes"""
        return self.buffer.get()
    
    def probabilities(self):
        """Gets the squared absolute value of each of the amplitudes"""
        out = pycl_array.to_device(
            self.queue,
            np.zeros(2**self.num_qubits, dtype=np.float32)
        )

        program.calculate_probabilities(
            self.queue,
            out.shape,
            None,
            self.buffer.data,
            out.data
        )

        return out.get()
        
    def release(self):
        self.buffer.base_data.release()
    
def load_binary_from_file(file_path, ctx):
    with open(file_path, "rb") as f:
        binary = f.read()
    return binary

def find_platform(pname):
    platforms = cl.get_platforms()
    for p in platforms:
        if p.name == pname:
            return p
    return None

def select_device(platform):
    devices = platform.get_devices(cl.device_type.ACCELERATOR)
    selected_device = os.getenv('PYOPENCL_DEVICE')
    if selected_device is None:
        return devices[0]
    try:
        index = int(selected_device)
        if index >= 0 and index < len(devices):
            return devices[index]
        raise ValueError("Invalid Device number")
    except Exception as e:
        print(str(e))
        print("PYOPENCL_DEVICE should be an integer", file=sys.stderr)

def get_program(context,device):
    path = Path(os.getenv('PYOPENCL_KERNEL'))
    if path is None:
        raise Exception("Please set the PYOPENCL_KERNEL variable with a valid kernel path")
    else:
        if not path.exists():
            raise Exception("Invalid path for PYOPENCL_KERNEL")
    binary_path = load_binary_from_file(path, context)
    return cl.Program(context,[device],[binary_path]).build()
        




def create_context():
    global context
    global program
    global platform
    global device

    platform = find_platform('Intel(R) FPGA SDK for OpenCL(TM)')
    device = select_device(platform)
    context = cl.Context([device])  # Create a context with the above device
    program = get_program(context, device)
