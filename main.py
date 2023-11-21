# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from ImageEncoding.QuantumImage import QuantumImage, test_image
from ImageEncoding.Encodings import FRQI
from  Filters.Filters import *

from tqdm import tqdm
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram, visualize_transition


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f"Hi, {name}")  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    print_hi("PyCharm")

    image = QuantumImage(test_image(side=8), zooming_factor=1)
    print(image.__info__())

    image.show_classical_image()
    print("Encoding")
    FRQI(image)
    print("Sobel")
    sobel(image)

    print("drawing circ")
    image.draw_circuit()
'''
    from qiskit.providers import aer
    from qiskit import execute, QuantumRegister
    from qiskit.tools.events import TextProgressBar
    import time

    qasm = aer.QasmSimulator(method="statevector")

    # In[ ]:

    numOfShots = 3000

    start_time = time.time()

    result = execute(
        image.circuit, qasm, shots=numOfShots
    ).result()  # ,,optimization_level=3
    # result = sim.run(circuit,shots=numOfShots,seed_simulator=12345).result()

    print("--- %s seconds ---" % (time.time() - start_time))
    print(len(result.get_counts(image.circuit)))

    # Create an empty array to save the retrieved image
    original = True
    counts = result.get_counts(image.circuit)
    print(counts)
    # plt.figure(2)
    # plot_histogram(counts)
    # plt.show()

    image.retrieve_and_show(result, numOfShots)
    # Create an empty array to save the retrieved image
    # retrieve_image = np.zeros((int(pow(2, image.required_qubits / 2)), int(pow(2, image.required_qubits / 2))))

    # We iterate over all the pixels and obtain the proability results for each one of them
    # Notice that we compute the ratio of the white states, and only if the values are 0
    # we make an exception and turn that pixel 0

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
'''