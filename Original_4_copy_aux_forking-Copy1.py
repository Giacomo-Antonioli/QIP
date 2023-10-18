#!/usr/bin/env python
# coding: utf-8

# In[1]:


generate=True


# In[2]:


#!pip install qiskit-terra[visualization]


# In[3]:


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



# In[4]:


theta = 0 # all pixels black
qc = QuantumCircuit(3)


qc.h(0)
qc.h(1)

qc.barrier()
#Pixel 1

qc.cry(theta,0,2)
qc.cx(0,1)
qc.cry(-theta,1,2)
qc.cx(0,1)
qc.cry(theta,1,2)

qc.barrier()
#Pixel 2

qc.x(1)
qc.cry(theta,0,2)
qc.cx(0,1)
qc.cry(-theta,1,2)
qc.cx(0,1)
qc.cry(theta,1,2)

qc.barrier()

qc.x(1)
qc.x(0)
qc.cry(theta,0,2)
qc.cx(0,1)
qc.cry(-theta,1,2)
qc.cx(0,1)
qc.cry(theta,1,2)


qc.barrier()

qc.x(1)

qc.cry(theta,0,2)
qc.cx(0,1)
qc.cry(-theta,1,2)
qc.cx(0,1)
qc.cry(theta,1,2)

qc.measure_all()
if display:
    qc.draw()


# aer_sim = Aer.get_backend('aer_simulator')
# t_qc = transpile(qc, aer_sim)
# qobj = assemble(t_qc, shots=4096)
# result = aer_sim.run(qobj).result()
# counts = result.get_counts(qc)
# print(counts)
# plot_histogram(counts)

# In[5]:


ar=np.asarray([0,240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,185,159,151,60,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,254,254,254,254,241,198,198,198,198,198,198,198,198,170,52,0,0,0,0,0,0,0,0,0,0,0,0,67,114,72,114,163,227,254,225,254,254,254,250,229,254,254,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,66,14,67,67,67,59,21,236,254,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,253,209,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,233,255,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,254,238,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,59,249,254,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,187,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,205,248,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,254,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,251,240,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,221,254,166,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,203,254,219,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,254,254,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,224,254,115,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,242,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,254,219,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,207,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#ar=np.asarray([0,100,200,255,0,100,200,255,0,100,200,255,0,100,200,255])

# Create an 8x8 matrix filled with zeros
#ar = np.zeros((4, 4))

# Define the size of the 4x4 square of ones
square_size = 2

# Calculate the starting and ending indices for the square
start_row = (4 - square_size) // 2
end_row = start_row + square_size
start_col = (4 - square_size) // 2
end_col = start_col + square_size

# Fill the square region with ones
#ar[start_row:end_row, start_col:end_col] = 255

#ar=np.transpose(ar.reshape((4,4)))
ar=ar.reshape((28,28))
#ar=ar[8:16,14:22]
ar=zoom(ar,0.25)
imagearray=copy.deepcopy(ar)
print("prePadding")
print(ar.size)
print(np.shape(ar))
nextval = pow(2, math.ceil(math.log(ar.size)/math.log(2)))

if nextval-ar.size>0:
    print("padded by")
    print(int(math.floor(math.log2(nextval-ar.size))/2))
    ar=np.pad(ar,(0,int(math.floor(math.log2(nextval-ar.size))/2)),'constant')

print("post padding")
print(ar.size)
print(np.shape(ar))
required_qubits=int(math.log2(np.shape(ar)[0]))*2
print(required_qubits)
print("Qubits n for x and y: " +str(int(required_qubits/2)))
intensity_qubits=8

total_qubits=int(required_qubits+1)
n_aux_qubit=4
num_summing=int(required_qubits/2)
num_carry= num_summing+1 
ar=np.abs(ar)
ar[ar > 255] = 255
ar[ar<50]=0
ar[4,3]=150
ar[7,3]=200
ar
plt.imshow(ar,cmap="gray")


# In[6]:


from tqdm import tqdm

def hadamard(circ, n):
    for i in n:
        circ.h(i)
        
def change(state, new_state):
    n = len(state)  # n is the length of the binary string
    c = np.array([])  # create an empty array
    for i in range(n):  # start to iterate n times
        if state[i] != new_state[i]:  # here we check if the position i in the 
                                  # binary string is different
                c = np.append(c, int(i))

    if len(c) > 0:
        return c.astype(int)
    else:
        return c

    

# This function applies the X-gates to the corresponding qubits given by n_aux_qubit
# the change function
def binary(circ, state, new_state):
  # state is the string in binary format that tells us the actual state of the pixels
  # state = '000000'
    c = change(state, new_state)
    if len(c) > 0:
    # Modified section. Added the -1 and the np.abs
        #print(c)
        #print(len(c))
        #input()
        circ.x(np.abs(c-(required_qubits+aux_qubit-1))) ##############################LOOOOK HERE
    else: 
        pass

    

# This function applies the C^n(Ry) (controlled Ry gate) in the qu antum circuit
# "circ" using the vector "n" for the controlled qubits and the variable "t" for
# the target qubit, and "theta" for the angle in the rotation. 
def cnri(circ, n, t, theta):
  #rq = circuit.qubits
    controls = len(n)
    cry = RYGate(2*theta).control(controls)
    aux = np.append(n, t).tolist()
    circ.append(cry, aux)
    

# The frqi function calls all of the aleady defined function to fully implement
# The FRQI method over the quantum circuit "circ", with the vector of controlled
# qubits "n", the target qubit "t" and the angles of each pixel in the image.

def frqi(circ,n ,t, angles):
  j = 0
  for i in tqdm(angles):
    state = '{0:013b}'.format(j-1)
    new_state = '{0:013b}'.format(j)
    
    if j == 0:
     
      cnri(circ, n, t, i)
    else:
     
      binary(circ, state, new_state)
      cnri(circ, n, t, i)
    j += 1
    circ.barrier()


# In[7]:


def xyfrqi(circ,n ,t, angles):
  j = 0
  bitsneeded='{0:0'+str(int(required_qubits/2))+'b}'
  print(bitsneeded)
  nqubits=circ.num_qubits
  print("TOTQUB: "+str(nqubits))
  print("requiredqubits:"+str(required_qubits))
  for index_row,rows in enumerate(tqdm(angles)):
        x_old=bitsneeded.format(index_row-1)[::-1]
        if index_row==0:
            changed=False
        else:
            changed=True
        
        for index_cols,i in enumerate((rows)):
            y=bitsneeded.format(index_cols)
           
            
            
            if index_row+index_cols>0:
                 if changed:
                    changed=False
                    x=bitsneeded.format(index_row)
                    tonegatex=[]
                    for index,element in enumerate(x[::-1]):
                        if element != x_old[index]:
                            #print("adding 1")
                            #print(nqubits+index-required_qubits/2-1)
                            tonegatex.append(int(nqubits+index-required_qubits/2-1-n_aux_qubit))
                            #print("tonegatx:"+str(int(nqubits+index-required_qubits/2-1-n_aux_qubit)))

                    circ.x(np.abs(tonegatex)+n_aux_qubit-(num_summing+num_carry))
                 tonegatey=[]
                 for index,element in enumerate(y[::-1]):
                    
                    if element != y_old[index]:
                       
                        tonegatey.append(int(nqubits+index-required_qubits-1-n_aux_qubit))
                        #print("tonegaty:"+str(int(nqubits+index-required_qubits-1-n_aux_qubit)))

                 circ.x(np.abs(tonegatey)+n_aux_qubit-(num_summing+num_carry))
            y_old=bitsneeded.format(index_cols)[::-1]
                    
                

            controls = len(n)
            cry = RYGate(2*i).control(controls)
            aux = np.append(n, t).tolist()
            circ.append(cry, aux)
            #circ.barrier()

normalized_pixels = ar/255.0
angles = np.arcsin(normalized_pixels)
print(np.shape(angles))
#angles=angles.reshape(1,ar.size)[0]



# In[8]:


def adder(circuit,wires,shift):
    
    num_summing=len(wires)
    num_carry= num_summing+1
    #if not visited:
    #    carryqubits=QuantumRegister(num_carry, 'carry')
    #    summing_val=QuantumRegister(num_summing,'shift')
    #    circuit.add_register(summing_val)
    #    circuit.add_register(carryqubits)
    #    visited=True
    
    number= format(shift, '0'+str(len(wires))+'b')
    print(number)
    for index,element in enumerate(number[::-1]):
        if element=='1':
            #print("SI, shift["+str(index)+"]")
            circuit.x(circuit.num_qubits-num_carry-num_summing+index)#da fare fuori
        
    for index,wire in enumerate(wires):
        
        carry_pre=circuit.num_qubits-num_carry+index
        #print("#0")
        #print(circuit.num_qubits)
        #print("#1")
        #print(num_carry)
        #print("#2")
        #print(wire)
        first=wire
        adder=circuit.num_qubits-num_carry-num_summing+index
        carry_post=carry_pre+1
        circuit.compose(carry, [carry_pre,first,adder,carry_post],inplace=True)  
        circuit.barrier()
    circuit.cx(wires[-1],circuit.num_qubits-num_carry-1)
    
    for index,wire in reversed(list(enumerate(wires))):
        
        carry_sum=circuit.num_qubits-(num_carry-index-1)-1
        add_1=wire
        add_2=circuit.num_qubits-num_carry-(num_summing-index)
        if index != len(wires)-1:
            carry_post=carry_sum+1
            circuit.compose(carry.inverse(),[carry_sum,add_1,add_2,carry_post],inplace=True)
            circuit.barrier()
        circuit.compose(sum, [carry_sum,add_1,add_2],inplace=True)  
    for index, wire in (list(enumerate(wires))):
        circuit.swap(wire,initial_qubits+index)
        circuit.reset(initial_qubits+index)
       
        circuit.barrier()


# In[9]:


#Qunatum sum gate
c = QuantumRegister(1, 'c')
a = QuantumRegister(1, 'a')
b = QuantumRegister(1, 'b')
# create the quantum circuit for the image
sum = QuantumCircuit(c,a,b,name="Sum")

# set the total number of qubits
num_qubits = sum.num_qubits
#single_bit_comparator.initialize('0011', single_bit_comparator.qubits)

#single_bit_comparator.initialize('1100', single_bit_comparator.qubits)

sum.cx(1,2)
sum.cx(0,2)

#sum.to_gate()
#sum.draw()
#Quantum carry gate
c = QuantumRegister(1, 'c')
a = QuantumRegister(1, 'a')
b = QuantumRegister(1, 'b')
d = QuantumRegister(1, 'c[i+1]')

# create the quantum circuit for the image
carry = QuantumCircuit(c,a,b,d,name="Carry")

# set the total number of qubits    return circuit

num_qubits = carry.num_qubits
#single_bit_comparator.initialize('0011', single_bit_comparator.qubits)

#single_bit_comparator.initialize('1100', single_bit_comparator.qubits)

carry.ccx(1,2,3)
carry.cx(1,2)
carry.ccx(0,2,3)
#carry.draw()
#carry.to_gate()
#carry.name="Carry"
#carry.label="Carry"
visited=False
original=QuantumRegister(num_summing, 'original')
shift=QuantumRegister(num_summing, 'shift')
carryqubits=QuantumRegister(num_carry, 'carry')
translation = QuantumCircuit(original,shift,carryqubits,name="Traslation")

for index in range(num_summing):

   translation.compose(carry, [index+2*num_summing,index,index+num_summing,index+2*num_summing+1],inplace=True)  
   #translation.cx(index,index+num_summing)
translation.cx(num_summing-1,2*num_summing-1)
for index in reversed(range(num_summing)):
    carry_sum=translation.num_qubits-(num_carry-index-1)-1
    add_1=index
    add_2=translation.num_qubits-num_carry-(num_summing-index)
    if index != num_summing-1:
        carry_post=carry_sum+1
        
        #inversecarrygate=carry.inverse().to_gate()
        translation.compose(carry.inverse(),[carry_sum,add_1,add_2,carry_post],inplace=True)
    translation.compose(sum, [carry_sum,add_1,add_2],inplace=True) 
for index in range(num_summing):
    translation.swap(index,num_summing+index)
    
controlled_translation=translation.to_gate().control(4)

from qiskit.tools.visualization import circuit_drawer



# In[10]:


x_qubits=QuantumRegister(required_qubits/2,'x')
y_qubits=QuantumRegister(required_qubits/2,'y')
aux_qubit=QuantumRegister(n_aux_qubit,'aux')
color_qubit=QuantumRegister(1,'c')
cr = ClassicalRegister(total_qubits+n_aux_qubit,'classical')

carryqubits=QuantumRegister(num_carry, 'carry')
summing_val=QuantumRegister(num_summing,'shift')
frqi_encoding = QuantumCircuit(aux_qubit,x_qubits,y_qubits,color_qubit,summing_val,carryqubits,cr)   
hadamard(frqi_encoding,range(total_qubits+n_aux_qubit-1) )


initial_qubits=frqi_encoding.num_qubits
# Does not modify original circuit
frqi_encoding.draw(scale=0.5)


# In[11]:


def encode_number(circuit,shift):
    number= format(shift, '0'+str(num_summing)+'b')
    for index,element in enumerate(number[::-1]):
        if element=='1':
            #print("SI, shift["+str(index)+"]")
            print("total qubits: "+str(total_qubits+n_aux_qubit))
            print(total_qubits+index)
            circuit.x(total_qubits+n_aux_qubit+index)#da fare fuori


def traslate_circuit(circuit,axis,shift):
    encode_number(circuit,shift)
    control_qubits=[i for i in range(n_aux_qubit)]
    if axis =='x':
        control_qubits=control_qubits+[i for i in range(n_aux_qubit,n_aux_qubit+num_summing)]
    elif axis=='y':
        control_qubits=control_qubits+[i for i in range(n_aux_qubit+num_summing,n_aux_qubit+2*num_summing)]
    control_qubits=control_qubits+[i for i in range(n_aux_qubit+total_qubits,n_aux_qubit+total_qubits+num_summing+num_carry)]
    frqi_encoding.append(controlled_translation,control_qubits)
    for index in range(num_summing):
        frqi_encoding.reset(n_aux_qubit+total_qubits+index)
      


# In[12]:


#traslate_circuit(frqi_encoding,'x',3)


# In[13]:


#frqi_encoding.draw()


# In[14]:


if generate:
    visited=False
    wires=[int(x) for x in range(n_aux_qubit,total_qubits+3)]
    print(wires)
    xyfrqi(frqi_encoding,wires,int(total_qubits+n_aux_qubit-1), angles)#1
    #tx=adder(frqi_encoding,wires[0:3],4)
    #tx.to_gate().control(4)
    #frqi_encoding.append(tx,wires[0:3]+[0,1,2,3])
    #shift=3
    traslate_circuit(frqi_encoding,'x',int(pow(2,int(required_qubits)/2)-1))
    traslate_circuit(frqi_encoding,'y',int(pow(2,int(required_qubits)/2)-1))
    #circuit_drawer(frqi_encoding,output='latex', fold=80,filename='./minicircuit.png')



# traslate_circuit(frqi_encoding,'x',3)
# traslate_circuit(frqi_encoding,'y',2)
# frqi_encoding.draw()
# 
# 
# 

# In[ ]:





# In[15]:


frqi_encoding.measure([int(x) for x in range(total_qubits+n_aux_qubit)],[int(x) for x in range(total_qubits+n_aux_qubit)])


# In[16]:


#frqi_encoding = frqi_encoding.decompose() # Does not modify original circuit
#frqi_encoding.draw()


# In[ ]:





#     frqi_encoding.x(total_qubits)
#     xyfrqi(frqi_encoding,wires,int(total_qubits+2), angles)#2
#     frqi_encoding.x(total_qubits+1)
#     xyfrqi(frqi_encoding,wires,int(total_qubits+2), angles)#3
#     frqi_encoding.x(total_qubits)
#     xyfrqi(frqi_encoding,wires,int(total_qubits+2), angles)#4
#     frqi_encoding.x(total_qubits)
#     frqi_encoding.x(total_qubits+1)
#     frqi_encoding.x(total_qubits+2)
#     xyfrqi(frqi_encoding,wires,int(total_qubits+2), angles)#5
#     frqi_encoding.x(total_qubits)
#     xyfrqi(frqi_encoding,wires,int(total_qubits+2), angles)#6
#     frqi_encoding.x(total_qubits)
#     frqi_encoding.x(total_qubits+1)
#     xyfrqi(frqi_encoding,wires,int(total_qubits+2), angles)#7
#     frqi_encoding.x(total_qubits)
#     xyfrqi(frqi_encoding,wires,int(total_qubits+2), angles)#8

# In[17]:


if generate:
    with open('4aux.qpy', 'wb') as fd:
        qpy.dump(frqi_encoding, fd)
else:
    with open('4aux.qpy', 'rb') as fd:
        frqi_encoding = qpy.load(fd)[0]

        
        


# In[18]:


from qiskit.tools.visualization import circuit_drawer
#circuit_drawer(frqi_encoding,output='latex_source', fold=80,filename='./minicircuit.tex')


# from multiprocessing import Pool
# import time
# import math
# import os
# from qiskit import execute, QuantumRegister
# from qiskit.tools.events import TextProgressBar
# import sys
# 
# cores=10
# numOfShots = 104 #8576
# singleShot=math.floor(numOfShots/cores)
# 
# total=[singleShot]*cores
# for i in range(0,numOfShots%cores):
#     total[i]+=1
#     
# def my_execute(current_circuit,shots):
#     print("Process ["+str(os.getpid())+"] started")
#     backend_sim = Aer.get_backend('qasm_simulator')
#     result = execute(current_circuit, backend_sim, shots=shots).result()   
#     return result
# 
# 
# from multiprocessing import Pool
# 
# print("start dupilcating struct")
# print("struct size: "+str(sys.getsizeof(frqi_encoding)))
# start_time = time.time()
# circuits=[frqi_encoding]*cores
# print("--- %s seconds ---" % (time.time() - start_time))
# start_time = time.time()
# if __name__ == '__main__':
#     print("Starting Pool")
#     with Pool(processes=cores) as pool:
#         results = pool.starmap(my_execute, zip(circuits,total))
#     print(results)
# 
# print("--- %s seconds ---" % (time.time() - start_time))
# 

# import sys
# print(result[0])
# print(sys.getsizeof(result))

# In[ ]:





# from qiskit import execute, QuantumRegister
# from qiskit.tools.events import TextProgressBar
# import time
# #sim = AerSimulator(method='statevector', device='GPU')
# 
# #circuit = transpile(frqi_encoding, sim)
# 
# backend_sim = Aer.get_backend('qasm_simulator')
# backend_sim.max_parallel_threads=72
# backend_sim.max_parallel_experiments=72
# 

# In[19]:


from qiskit.providers import aer
from qiskit import execute, QuantumRegister
from qiskit.tools.events import TextProgressBar
import time
qasm=aer.QasmSimulator(method="statevector")


# In[ ]:


numOfShots = 20000

start_time = time.time()

result = execute(frqi_encoding, qasm, shots=numOfShots).result()#,,optimization_level=3
#result = sim.run(circuit,shots=numOfShots,seed_simulator=12345).result()

print("--- %s seconds ---" % (time.time() - start_time))
print(len(result.get_counts(frqi_encoding)))


# -16 sec
# 

# In[ ]:


if display:
    print(result)


# In[ ]:


print(result.get_counts(frqi_encoding)) 


# In[ ]:


print(sorted(result.get_counts(frqi_encoding)))


# In[ ]:


print(len(result.get_counts(frqi_encoding)))


# In[ ]:


test=result.get_counts(frqi_encoding)
counttest=0
secondtest=0
for i in test:
    if i[0]=='1':
        counttest+=1
        if test[i]>8000:
            secondtest+=1
print(counttest)
print(secondtest)


# In[ ]:


# (Optional) Print the results of the measurements as histogram
#plot_histogram(result.get_counts(frqi_encoding), figsize=(20,11))
print("SKIPPED")


# In[ ]:


# Create an empty array to save the retrieved image
print(required_qubits)
retrieve_image_0 =np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_1 = np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_2 =np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_3 = np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_4 = np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_5 = np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_6 = np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_7 = np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_8 =np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_9 = np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_10 =np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_11 = np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_12 = np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_13 = np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_14 = np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))
retrieve_image_15 = np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))

# We iterate over all the pixels and obtain the proability results for each one of them
# Notice that we compute the ratio of the white states, and only if the values are 0
# we make an exception and turn that pixel 0
original=True

# Create an empty array to save the retrieved image
retrieve_image = np.zeros((int(pow(2,required_qubits/2)),int(pow(2,required_qubits/2))))

# We iterate over all the pixels and obtain the proability results for each one of them
# Notice that we compute the ratio of the white states, and only if the values are 0
# we make an exception and turn that pixel 0
original=True
count=0
count_1=0
print(required_qubits)
with tqdm(total=pow(2,n_aux_qubit)*pow(2,required_qubits)) as pbar:
    for i in (range(pow(2,n_aux_qubit)*pow(2,required_qubits))):
      try:
        bitsneeded='{0:0'+str(int(required_qubits)+n_aux_qubit)+'b}'
        s = bitsneeded.format(i)
        new_s = '1' + s
        #print("S: "+s)
        
        extracted_value=np.sqrt(result.get_counts(frqi_encoding)[new_s]/numOfShots)
        y=s[:int(required_qubits/2)]
        x=s[int(required_qubits/2):required_qubits]
        #print("[3:"+str(int(required_qubits/2)+3)+"]["+str(int(required_qubits/2)+3)+":"+str(int(required_qubits)+3)+"]")
        x_index=0
        y_index=0

        for index in range(int(required_qubits/2)):

            if x[-index-1]=='1':
                y_index += pow(2,index)
            if y[-index-1]=='1':
                x_index += pow(2,index)
        #print("Y_INDEX: "+str(y_index))
        #print("X_INDEX: "+str(x_index))
                #print("IMAGE "+s[-4:-1]+" AT: ["+str(x_index)+","+str(y_index)+"]:"+str(8*extracted_value))

        
        match s[-n_aux_qubit:]:
            case '0000':
                retrieve_image_0[int(x_index)][int(y_index)] = extracted_value
            case '0001':
                retrieve_image_1[int(x_index)][int(y_index)] = extracted_value
            case '0010':
                retrieve_image_2[int(x_index)][int(y_index)] = extracted_value
            case '0011':
                retrieve_image_3[int(x_index)][int(y_index)] = extracted_value
            case '0100':
                retrieve_image_4[int(x_index)][int(y_index)] = extracted_value
            case '0101':
                retrieve_image_5[int(x_index)][int(y_index)] = extracted_value
            case '0110':
                retrieve_image_6[int(x_index)][int(y_index)] = extracted_value
            case '0111':
                retrieve_image_7[int(x_index)][int(y_index)] = extracted_value
            case '1000':
                retrieve_image_8[int(x_index)][int(y_index)] = extracted_value
            case '1001':
                retrieve_image_9[int(x_index)][int(y_index)] = extracted_value
            case '1010':
                retrieve_image_10[int(x_index)][int(y_index)] = extracted_value
            case '1011':
                retrieve_image_11[int(x_index)][int(y_index)] = extracted_value
            case '1100':
                retrieve_image_12[int(x_index)][int(y_index)] = extracted_value
            case '1101':
                retrieve_image_13[int(x_index)][int(y_index)] = extracted_value
            case '1110':
                retrieve_image_14[int(x_index)][int(y_index)] = extracted_value
            case '1111':
                retrieve_image_15[int(x_index)][int(y_index)] = extracted_value

        count+=1
        pbar.update(1)
        #print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°˚")
      except KeyError:
        pbar.update(1)
        pass
print(count)


# In[ ]:


128/8


# In[ ]:


plt.figure(1)
multip=32
plt.subplot(241)

retrieve_image_0 *=  multip*255.0
retrieve_image_0 = retrieve_image_0.astype('int')
retrieve_image_0 = retrieve_image_0.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_0, cmap='gray', vmin=0, vmax=255)

plt.subplot(242)

retrieve_image_1 *=  multip*255.0
retrieve_image_1 = retrieve_image_1.astype('int')
retrieve_image_1 = retrieve_image_1.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_1, cmap='gray', vmin=0, vmax=255)

plt.subplot(243)

retrieve_image_2 *=  multip*255.0
retrieve_image_2 = retrieve_image_2.astype('int')
retrieve_image_2 = retrieve_image_2.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_2, cmap='gray', vmin=0, vmax=255)

plt.subplot(244)

retrieve_image_3 *=  multip*255.0
retrieve_image_3 = retrieve_image_3.astype('int')
retrieve_image_3 = retrieve_image_3.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_3, cmap='gray', vmin=0, vmax=255)

plt.subplot(245)


retrieve_image_4 *=  multip*255.0
retrieve_image_4 = retrieve_image_4.astype('int')
retrieve_image_4 = retrieve_image_4.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_4, cmap='gray', vmin=0, vmax=255)

plt.subplot(246)

retrieve_image_5 *=  multip*255.0
retrieve_image_5 = retrieve_image_5.astype('int')
retrieve_image_5 = retrieve_image_5.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_5, cmap='gray', vmin=0, vmax=255)

plt.subplot(247)

retrieve_image_6 *=  multip*255.0
retrieve_image_6 = retrieve_image_6.astype('int')
retrieve_image_6 = retrieve_image_6.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_6, cmap='gray', vmin=0, vmax=255)

plt.subplot(248)

retrieve_image_8 *=  multip*255.0
retrieve_image_8 = retrieve_image_0.astype('int')
retrieve_image_8 = retrieve_image_0.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_8, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.figure(2)
plt.subplot(241)

retrieve_image_9 *=  multip*255.0
retrieve_image_9 = retrieve_image_9.astype('int')
retrieve_image_9 = retrieve_image_9.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_9, cmap='gray', vmin=0, vmax=255)

plt.subplot(242)

retrieve_image_10 *=  multip*255.0
retrieve_image_10 = retrieve_image_10.astype('int')
retrieve_image_10 = retrieve_image_10.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_10, cmap='gray', vmin=0, vmax=255)

plt.subplot(243)

retrieve_image_11 *=  multip*255.0
retrieve_image_11 = retrieve_image_11.astype('int')
retrieve_image_11 = retrieve_image_11.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_11, cmap='gray', vmin=0, vmax=255)

plt.subplot(244)


retrieve_image_12 *=  multip*255.0
retrieve_image_12 = retrieve_image_12.astype('int')
retrieve_image_12 = retrieve_image_12.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_12, cmap='gray', vmin=0, vmax=255)

plt.subplot(245)

retrieve_image_13 *=  multip*255.0
retrieve_image_13 = retrieve_image_13.astype('int')
retrieve_image_13 = retrieve_image_13.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_13, cmap='gray', vmin=0, vmax=255)

plt.subplot(246)

retrieve_image_14 *=  multip*255.0
retrieve_image_14 = retrieve_image_14.astype('int')
retrieve_image_14 = retrieve_image_14.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_14, cmap='gray', vmin=0, vmax=255)

plt.subplot(247)

retrieve_image_15 *=  multip*255.0
retrieve_image_15 = retrieve_image_15.astype('int')
retrieve_image_15 = retrieve_image_15.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_15, cmap='gray', vmin=0, vmax=255)
plt.subplot(248)
retrieve_image_7 *=  multip*255.0
retrieve_image_7 = retrieve_image_7.astype('int')
retrieve_image_7 = retrieve_image_7.reshape((pow(2,int(required_qubits/2)),pow(2,int(required_qubits/2))))
# Plot the retrieved image to see if it is the same as the one encoded
plt.imshow(retrieve_image_7, cmap='gray', vmin=0, vmax=255)




plt.show()
plt.imshow(ar)


# In[ ]:


9*8


# In[ ]:


ar


# In[ ]:


retrieve_image_0


# In[ ]:


print("CIRCUIT STATS FRQI")
print("width: "+str(frqi_encoding.width()))
print("depth: "+str(frqi_encoding.depth()))
print("size: "+str(frqi_encoding.size()))
print("ops: "+str(frqi_encoding.count_ops()))


# In[ ]:





# 
# for i in range(8*pow(2,required_qubits)):
#     try:
#     
#         s = format(i, '013b')
#         new_s = '1' + s
#         match new_s[1:4]:
#             case '000':
#                 retrieve_image_0 = np.append(retrieve_image_0,np.sqrt(result.get_counts(frqi_encoding)[new_s]/numOfShots))
#             case '001':
#                 retrieve_image_1 = np.append(retrieve_image_1,np.sqrt(result.get_counts(frqi_encoding)[new_s]/numOfShots))
#             case '010':
#                 retrieve_image_2 = np.append(retrieve_image_2,np.sqrt(result.get_counts(frqi_encoding)[new_s]/numOfShots))
#             case '011':
#                 retrieve_image_3 = np.append(retrieve_image_3,np.sqrt(result.get_counts(frqi_encoding)[new_s]/numOfShots))
#             case '100':
#                 retrieve_image_4 = np.append(retrieve_image_4,np.sqrt(result.get_counts(frqi_encoding)[new_s]/numOfShots))
#             case '101':
#                 retrieve_image_5 = np.append(retrieve_image_5,np.sqrt(result.get_counts(frqi_encoding)[new_s]/numOfShots))
#             case '110':
#                 retrieve_image_6 = np.append(retrieve_image_6,np.sqrt(result.get_counts(frqi_encoding)[new_s]/numOfShots))
#             case '111':
#                 retrieve_image_7 = np.append(retrieve_image_7,np.sqrt(result.get_counts(frqi_encoding)[new_s]/numOfShots))
# 
#     except KeyError:
# 
#         match new_s[1:4]:
#             case '000':
#                 retrieve_image_0 = np.append(retrieve_image_0,[0.0])
#             case '001':
#                 retrieve_image_1 = np.append(retrieve_image_1,[0.0])
#             case '010':
#                 retrieve_image_2 = np.append(retrieve_image_2,[0.0])
#             case '011':
#                 retrieve_image_3 = np.append(retrieve_image_3,[0.0])
#             case '100':
#                 retrieve_image_4 = np.append(retrieve_image_4,[0.0])
#             case '101':
#                 retrieve_image_5 = np.append(retrieve_image_5,[0.0])
#             case '110':
#                 retrieve_image_6 = np.append(retrieve_image_6,[0.0])
#             case '111':
#                 retrieve_image_7 = np.append(retrieve_image_7,[0.0])
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:




