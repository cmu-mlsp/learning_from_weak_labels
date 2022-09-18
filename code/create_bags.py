import gzip
import struct
import numpy as np
import pdb
import itertools
import numpy as np
from scipy import io

with gzip.open('t10k-images-idx3-ubyte.gz','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
    data_test = data.reshape((size, nrows, ncols))
with gzip.open('train-images-idx3-ubyte.gz','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
    data_train = data.reshape((size, nrows, ncols))
with gzip.open('train-labels-idx1-ubyte.gz','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    #nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
    data_train_labels = data.reshape((size,))
with gzip.open('t10k-labels-idx1-ubyte.gz','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    #nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
    data_test_labels = data.reshape((size,))

import random
random.seed(2022)
def Rand(start,end,num):
	res = []
	for j in range(num):
		res.append(random.randint(start,end))
	return res

random_train_ids = Rand(0,49999,40000)

data_train_subset = []
data_train_labels_subset = []
data_val = []
data_val_labels = []
for i in range(50000):
	if i in random_train_ids:
		data_train_subset.append(data_train[i])
		data_train_labels_subset.append(data_train_labels[i])
	else:
		data_val.append(data_train[i])
		data_val_labels.append(data_train_labels[i])

id_2_train = [i for i,x in enumerate(data_train_labels_subset) if x==2]
id_2_val = [i for i,x in enumerate(data_val_labels) if x==2]
id_2_test = [i for i,x in enumerate(data_test_labels) if x==2]

id_neg_train = [i for i,x in enumerate(data_train_labels_subset) if x!=2]
id_neg_val = [i for i,x in enumerate(data_val_labels) if x!=2]
id_neg_test = [i for i,x in enumerate(data_test_labels) if x!=2]

bag_size=20
max_pos_samples = 4
num_bags = 100
num_bags_val = 50

def generate_pos_bags(num_bags,max_pos_samples,bag_size,pos_ids,neg_ids,data,data_labels):
	np.random.seed(2022)
	data_pos_bag = []
	data_pos_bag_label = []
	pos_sample_count_in_each_bag = np.random.randint(1,max_pos_samples,num_bags).tolist()	
	for i in range(num_bags):
		bag = []
		bag_labels = []
		#num_pos = random.sample(pos_samples,1)[0]
		pos_sample_id = np.random.choice(pos_ids,pos_sample_count_in_each_bag[i]).tolist()
		for id in pos_sample_id:
			bag.append(list(np.array(data[id]).flatten(order='F')))
			bag_labels.append(data_labels[id])
		num_neg = bag_size - pos_sample_count_in_each_bag[i]
		neg_sample_id = np.random.choice(neg_ids,num_neg).tolist()
		for id in neg_sample_id:
			bag.append(list(np.array(data[id]).flatten(order='F')))
			bag_labels.append(data_labels[id])
		data_pos_bag.append(bag)
		data_pos_bag_label.append(bag_labels)
	return data_pos_bag,data_pos_bag_label


def generate_neg_bags(num_bags,bag_size,neg_ids,data,data_labels):
	np.random.seed(2022)
	data_neg_bag = []
	data_neg_bag_label = []
	for i in range(num_bags):
		bag = []
		bag_labels = []
		num_neg = bag_size
		neg_sample_id = np.random.choice(neg_ids,num_neg).tolist()
		for id in neg_sample_id:
			bag.append(list(np.array(data[id]).flatten(order='F')))
			bag_labels.append(data_labels[id])
		data_neg_bag.append(bag)
		data_neg_bag_label.append(bag_labels)
	return data_neg_bag,data_neg_bag_label


# generate positive train bags
data_pos_bag_train,data_pos_bag_train_label = generate_pos_bags(num_bags,max_pos_samples,bag_size,id_2_train,id_neg_train,data_train_subset,data_train_labels_subset)

# generate positive val bags
data_pos_bag_val,data_pos_bag_val_label = generate_pos_bags(num_bags_val,max_pos_samples,bag_size,id_2_val,id_neg_val,data_val,data_val_labels)

# generate positive test bags
data_pos_bag_test,data_pos_bag_test_label = generate_pos_bags(num_bags,max_pos_samples,bag_size,id_2_test,id_neg_test,data_test,data_test_labels)

# generate negative train bags
data_neg_bag_train,data_neg_bag_train_label = generate_neg_bags(num_bags,bag_size,id_neg_train,data_train_subset,data_train_labels_subset)

# generate negative val bags
data_neg_bag_val,data_neg_bag_val_label = generate_neg_bags(num_bags,bag_size,id_neg_val,data_val,data_val_labels)

# generate negative test_bags
data_neg_bag_test,data_neg_bag_test_label = generate_neg_bags(num_bags,bag_size,id_neg_test,data_test,data_test_labels)

io.savemat('test_try1.mat',{'pos_bags':data_pos_bag_test,'pos_bags_labels':data_pos_bag_test_label,'neg_bags':data_neg_bag_test,'neg_bags_labels':data_neg_bag_test_label})
io.savemat('train_try1.mat',{'pos_bags':data_pos_bag_train,'pos_bags_labels':data_pos_bag_train_label,'neg_bags':data_neg_bag_train,'neg_bags_labels':data_neg_bag_train_label})
io.savemat('val_try1.mat',{'pos_bags':data_pos_bag_val,'pos_bags_labels':data_pos_bag_val_label,'neg_bags':data_neg_bag_val,'neg_bags_labels':data_neg_bag_val_label})

pdb.set_trace()






############################################
####### END OF FILE
############################################

## generate positive train_bags
#data_pos_bag_train = []
#data_pos_bag_train_label = []
#for i in range(num_bags):
#        bag = []
#        bag_labels = []
#        num_pos = random.sample(pos_samples,1)[0]
#        pos_sample_id = random.sample(id_2_train,num_pos)
#        for id in pos_sample_id:
#                bag.append(list(np.array(data_train[id]).flatten(order='F')))
#                bag_labels.append(data_train_labels[id])
#        num_neg = bag_size - num_pos
#        neg_sample_id = random.sample(id_neg_train,num_neg)
#        for id in neg_sample_id:
#                bag.append(list(np.array(data_train[id]).flatten(order='F')))
#                bag_labels.append(data_train_labels[id])
#        data_pos_bag_train.append(bag)
#        data_pos_bag_train_label.append(bag_labels)

## generate negative train_bags
#data_neg_bag_train = []
#data_neg_bag_train_label = []
#for i in range(num_bags):
#        bag = []
#        bag_labels = []
#        num_neg = bag_size 
#        neg_sample_id = random.sample(id_neg_train,num_neg)
#        for id in neg_sample_id:
#                bag.append(list(np.array(data_train[id]).flatten(order='F')))
#                bag_labels.append(data_train_labels[id])
#        data_neg_bag_train.append(bag)
#        data_neg_bag_train_label.append(bag_labels)

## generate negative test_bags
#data_neg_bag_test = []
#data_neg_bag_test_label = []
#for i in range(num_bags):
#        bag = []
#        bag_labels = []
#        num_neg = bag_size 
#        neg_sample_id = random.sample(id_neg_test,num_neg)
#        for id in neg_sample_id:
#                bag.append(list(np.array(data_test[id]).flatten(order='F')))
#                bag_labels.append(data_test_labels[id])
#        data_neg_bag_test.append(bag)
#        data_neg_bag_test_label.append(bag_labels)

# generate positive test_bags
#data_pos_bag_test = []
#data_pos_bag_test_label = []
#for i in range(num_bags):
#        bag = []
#        bag_labels = []
#        num_pos = random.sample(pos_samples,1)[0]
#        pos_sample_id = random.sample(id_2_test,num_pos)
#        for id in pos_sample_id:
#                bag.append(list(np.array(data_test[id]).flatten(order='F')))
#                bag_labels.append(data_test_labels[id])
#        num_neg = bag_size - num_pos
#        neg_sample_id = random.sample(id_neg_test,num_neg)
#        for id in neg_sample_id:
#                bag.append(list(np.array(data_test[id]).flatten(order='F')))
#                bag_labels.append(data_test_labels[id])
#        data_pos_bag_test.append(bag)
#        data_pos_bag_test_label.append(bag_labels)

