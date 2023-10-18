import qiskit as qk
from qiskit import QuantumCircuit, Aer, IBMQ
from qiskit import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import plot_histogram,visualize_transition
from math import pi
import numpy as np
import copy
from PIL import Image
from numpy import array
import matplotlib.pyplot as plt
import math
import copy
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library.standard_gates import RYGate, RYYGate
from qiskit import qpy

display=False
from scipy.ndimage import zoom


